"""
model_loader.py - Hardware-aware model selection and inference speedups.

Picks LLM / reranker / whisper model sizes based on available RAM so the
same codebase runs on a 4GB low-end box up to a 12GB+ box, with a dedicated
Apple Silicon tier that accounts for unified memory pressure. Also provides:
  - OptimizedCLIPEncoder: ONNX INT8 CLIP text encoder (falls back to plain
    sentence-transformers CLIP if the ONNX export/load path fails)
  - upgrade_faiss_index: converts a flat FAISS index to IVF for faster search

Apple Silicon note: On M-series chips RAM is unified (CPU + GPU share the same
pool). With OS + Python + Redis the effective headroom for a model on an 8GB
machine is ~4-5GB. moondream2 (~1.8GB quantized) is the lightest vision-capable
Ollama model and runs well under Metal without swapping. num_ctx=2048 caps the
KV cache so the model load stays predictable.
"""

import logging
import os
import platform

import numpy as np
import psutil

logger = logging.getLogger(__name__)

CLIP_ONNX_DIR = "./models/clip-onnx"
CLIP_MODEL_NAME = "clip-ViT-B-32"


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_hardware_profile() -> dict:
    """Return {ram_gb, has_gpu, gpu_vram_gb, cpu_cores, apple_silicon} for the current machine."""
    ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
    cpu_cores = psutil.cpu_count(logical=True) or 1

    has_gpu = False
    gpu_vram_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            has_gpu = True
            gpu_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except ImportError:
        pass

    return {
        "ram_gb": ram_gb,
        "has_gpu": has_gpu,
        "gpu_vram_gb": gpu_vram_gb,
        "cpu_cores": cpu_cores,
        "apple_silicon": _is_apple_silicon(),
    }


def load_model_config() -> dict:
    """Auto-select LLM / reranker / whisper / worker-count based on available RAM.

    Environment overrides (set before running python app.py):
      RAG_MODEL=llava:7b          — force a specific model regardless of RAM
      RAG_NUM_CTX=1024            — override context window size

    Running llava:7b on Apple M3 8 GB (for benchmarking / compare_models.py):
      1. Stop Ollama.
      2. Start it with flash attention to cut KV-cache memory ~35%:
             OLLAMA_FLASH_ATTENTION=1 ollama serve
      3. Pull the model if you haven't:
             ollama pull llava:7b
      4. Run the comparison:
             RAG_MODEL=llava:7b RAG_NUM_CTX=1024 python compare_models.py
      Flash attention + num_ctx=1024 together bring llava:7b's working memory
      to ~4.8 GB on M3, which fits alongside the OS + Python stack on 8 GB.
    """
    profile = get_hardware_profile()
    ram_gb = profile["ram_gb"]
    apple_silicon = profile["apple_silicon"]

    # Environment overrides — used by compare_models.py to benchmark
    # alternative models without touching code.
    env_model = os.environ.get("RAG_MODEL")
    env_ctx = os.environ.get("RAG_NUM_CTX")

    # Apple Silicon (M1/M2/M3/M4) with ≤10 GB unified memory.
    # qwen2.5vl:3b (~2.3 GB Q4_K_M) is the best accuracy/memory tradeoff at
    # this size — significantly outperforms moondream2 on document QA, uses
    # less memory than llava-phi3, and runs well under Metal (MPS). num_ctx
    # stays at 2048 to keep the KV-cache small; workers at 1 to avoid
    # concurrent Whisper/CLIP loads overloading unified memory.
    if apple_silicon and ram_gb <= 10:
        mode = "apple_silicon_low"
        config = {
            "llm": "qwen2.5vl:3b",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-2-v2",
            "whisper": "tiny",
            "workers": 1,
            "num_ctx": 2048,
        }
    elif ram_gb >= 12:
        mode = "high"
        config = {
            "llm": "llava:7b",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "whisper": "base",
            "workers": 8,
            "num_ctx": 4096,
        }
    elif ram_gb >= 6:
        mode = "medium"
        config = {
            # llava-phi3: ~2.9GB, Phi-3-based, vision-capable, quantized by
            # default via Ollama. Good quality for the 6-12GB tier.
            "llm": "llava-phi3",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-2-v2",
            "whisper": "base",
            # Kept at 2: each parallel worker loads its own copy of
            # Whisper/CLIP, so more workers overload 6-9GB machines.
            "workers": 2,
            "num_ctx": 2048,
        }
    else:
        mode = "low"
        config = {
            "llm": "phi3:mini",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-2-v2",
            "whisper": "tiny",
            "workers": 2,
            "num_ctx": 2048,
        }

    # Apply env var overrides last so compare_models.py can benchmark any model.
    if env_model:
        config["llm"] = env_model
    if env_ctx:
        config["num_ctx"] = int(env_ctx)

    apple_note = " (Apple Silicon / unified memory)" if apple_silicon else ""
    print(f"[model_loader] Detected {ram_gb}GB RAM{apple_note} -> {mode.upper()} mode: {config}")
    return config


class OptimizedCLIPEncoder:
    """CLIP text/image encoder, ONNX INT8-accelerated when possible.

    Tries to load a pre-exported ONNX model from CLIP_ONNX_DIR; if that's
    missing, tries to export one with `optimum`; if the export also fails
    (no internet, incompatible optimum version, etc.), falls back to the
    plain sentence-transformers CLIP model so the pipeline keeps working.
    """

    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        self.model_name = model_name
        self.backend = None  # "onnx" or "sentence-transformers"
        self._model = None
        self._tokenizer = None

        if self._try_load_onnx():
            self.backend = "onnx"
            logger.info("OptimizedCLIPEncoder: using ONNX INT8 backend")
            return

        if self._try_export_onnx():
            self.backend = "onnx"
            logger.info("OptimizedCLIPEncoder: exported and loaded ONNX INT8 backend")
            return

        self._load_sentence_transformers_fallback()
        self.backend = "sentence-transformers"
        logger.warning("OptimizedCLIPEncoder: ONNX unavailable, falling back to plain CLIP")

    def _try_load_onnx(self) -> bool:
        if not os.path.isdir(CLIP_ONNX_DIR):
            return False
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer

            self._model = ORTModelForFeatureExtraction.from_pretrained(CLIP_ONNX_DIR)
            self._tokenizer = AutoTokenizer.from_pretrained(CLIP_ONNX_DIR)
            return True
        except Exception as e:
            logger.debug("OptimizedCLIPEncoder: failed loading existing ONNX dir (%s)", e)
            return False

    def _try_export_onnx(self) -> bool:
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
            from transformers import AutoTokenizer

            os.makedirs(CLIP_ONNX_DIR, exist_ok=True)
            model = ORTModelForFeatureExtraction.from_pretrained(self.model_name, export=True)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            quantizer = ORTQuantizer.from_pretrained(model)
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
            quantizer.quantize(save_dir=CLIP_ONNX_DIR, quantization_config=qconfig)

            tokenizer.save_pretrained(CLIP_ONNX_DIR)
            self._model = ORTModelForFeatureExtraction.from_pretrained(CLIP_ONNX_DIR)
            self._tokenizer = tokenizer
            return True
        except Exception as e:
            logger.debug("OptimizedCLIPEncoder: ONNX export failed (%s)", e)
            return False

    def _load_sentence_transformers_fallback(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list) -> np.ndarray:
        """Encode a list of strings into an (n, dim) array of embeddings."""
        if self.backend == "onnx":
            import torch
            inputs = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.numpy()

        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def upgrade_faiss_index(flat_index, all_embeddings: np.ndarray):
    """Convert a flat FAISS index into a trained IVFFlat index (nlist=100,
    nprobe=10) for faster approximate search on larger corpora."""
    import faiss

    d = flat_index.d
    quantizer = faiss.IndexFlatL2(d)
    ivf_index = faiss.IndexIVFFlat(quantizer, d, 100)
    ivf_index.train(all_embeddings)
    ivf_index.add(all_embeddings)
    ivf_index.nprobe = 10
    return ivf_index


CONFIG = load_model_config()
