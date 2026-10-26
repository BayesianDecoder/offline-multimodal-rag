#!/usr/bin/env python3
"""
compare_models.py — head-to-head evaluation of two Ollama models on the
existing legal RAG index.

Usage:
    python compare_models.py
    python compare_models.py --model-a qwen2.5vl:3b --model-b llava:7b
    python compare_models.py --queries custom_queries.json

Prereqs:
    • Processed FAISS index must exist at processed_output/multimodal_index.bin
      (run app.py and upload files first, or it may already exist).
    • Ollama must be running: ollama serve
    • Both models must be pulled:
          ollama pull qwen2.5vl:3b
          ollama pull llava:7b
    • For llava:7b on Apple M3 8GB — enable flash attention to reduce
      KV-cache memory ~35% so the 7B fits alongside the OS:
          OLLAMA_FLASH_ATTENTION=1 ollama serve    # instead of plain `ollama serve`
      Then keep num_ctx ≤ 1024 (already set by --model-b-ctx below).

Output:
    • Terminal: per-query comparison table
    • comparison_results.json: full results including per-sentence breakdowns
    • comparison_report.html: styled side-by-side HTML report
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# ── default test battery ────────────────────────────────────────────────────
# Covers all four modalities plus an out-of-domain probe.
DEFAULT_QUERIES = [
    {
        "id": "bns_sections",
        "query": "What are the key sections and punishments defined in the Bharatiya Nyaya Sanhita 2023?",
        "category": "PDF — statute",
        "expected_grounded": True,
    },
    {
        "id": "fir_incident",
        "query": "What was the alleged assault incident described in the FIR filed in Varanasi?",
        "category": "DOCX — FIR",
        "expected_grounded": True,
    },
    {
        "id": "bail_grounds",
        "query": "What are the grounds for bail stated in the bail application from the Sessions Court Lucknow?",
        "category": "DOCX — bail",
        "expected_grounded": True,
    },
    {
        "id": "employment_notice",
        "query": "What is the notice period stated in the employment agreement?",
        "category": "DOCX — contract",
        "expected_grounded": True,
    },
    {
        "id": "ncrb_stats",
        "query": "What were the total crimes against women reported in India in 2023 according to NCRB?",
        "category": "PDF + image — stats",
        "expected_grounded": True,
    },
    {
        "id": "court_hierarchy",
        "query": "How is the Indian court system structured from the Supreme Court down to lower courts?",
        "category": "image — diagram",
        "expected_grounded": True,
    },
    {
        "id": "legal_rights_arrest",
        "query": "What are your legal rights when you are arrested in India?",
        "category": "PDF — rights guide",
        "expected_grounded": True,
    },
    {
        "id": "data_privacy",
        "query": "What personal data processing violations are alleged in the DPDP PIL?",
        "category": "DOCX — PIL",
        "expected_grounded": True,
    },
    {
        "id": "out_of_domain",
        "query": "What is the capital of France?",
        "category": "out-of-domain probe",
        "expected_grounded": False,
    },
    {
        "id": "sc_aadhaar",
        "query": "What was the Supreme Court's ruling in the Ratan Lal case regarding Aadhaar and voter ID linkage?",
        "category": "PDF — SC judgment",
        "expected_grounded": True,
    },
]

FAISS_INDEX_PATH = "processed_output/multimodal_index.bin"
RESULTS_JSON = "comparison_results.json"
RESULTS_HTML = "comparison_report.html"


# ── helpers ─────────────────────────────────────────────────────────────────

def _patch_ollama_model(model_name: str, num_ctx: int):
    """Monkey-patch ragchatbot's module-level OLLAMA_MODEL / OLLAMA_NUM_CTX
    so the existing generate_answer() picks up the new model without
    reloading the FAISS index or the embedding model."""
    import ragchatbot as rc
    rc.OLLAMA_MODEL = model_name
    rc.OLLAMA_NUM_CTX = num_ctx


def _run_single_query(chatbot, query: str, model_name: str, num_ctx: int) -> dict:
    """Run one query through the pipeline and return timing + eval scores."""
    from src.cache.redis_cache import RAGCache
    from src.evaluation.hallucination_eval import (
        evaluate_response,
        check_retrieval_confidence,
    )

    _patch_ollama_model(model_name, num_ctx)

    cache = RAGCache.__new__(RAGCache)
    cache.r = None  # run without Redis for clean comparison

    t0 = time.time()

    # Retrieval
    results = chatbot.search_by_text(query, k=15)

    if not results:
        return {
            "answer": "No results found.",
            "latency_ms": round((time.time() - t0) * 1000),
            "declined": True,
            "eval_scores": None,
        }

    # Confidence gate
    confidence = check_retrieval_confidence(
        query,
        [r["metadata"].get("content", "") for r in results],
        chatbot.search_engine.reranker,
    )
    if not confidence["confident"]:
        return {
            "answer": "I don't have relevant information about that in the indexed documents.",
            "latency_ms": round((time.time() - t0) * 1000),
            "declined": True,
            "confidence_score": confidence["best_score"],
            "eval_scores": None,
        }

    # Generation
    answer = chatbot.generate_answer(query, results, query_type="text")
    latency_ms = round((time.time() - t0) * 1000)

    # Hallucination eval
    chunk_texts = [r["metadata"].get("content", "") for r in results]
    try:
        eval_scores = evaluate_response(
            query, answer, chunk_texts,
            chatbot.generator, chatbot.search_engine.reranker,
        )
    except Exception as e:
        print(f"  [eval error: {e}]")
        eval_scores = None

    return {
        "answer": answer,
        "latency_ms": latency_ms,
        "declined": False,
        "confidence_score": confidence["best_score"],
        "eval_scores": eval_scores,
        "sources": [
            r["metadata"].get("source", "?") for r in results[:3]
        ],
    }


# ── main ────────────────────────────────────────────────────────────────────

def _run_all_queries(chatbot, queries, model, ctx, label):
    """Run all queries with one model in a single pass (avoids model swapping on
    low-memory machines — Ollama only loads the model once this way)."""
    print(f"\n{'='*70}")
    print(f"  PASS: {label} ({model}, ctx={ctx})")
    print(f"{'='*70}")
    results = {}
    for i, q in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {q['id']}: {q['query'][:70]}…" if len(q['query'])>70 else f"  [{i}/{len(queries)}] {q['id']}: {q['query']}")
        r = _run_single_query(chatbot, q["query"], model, ctx)
        status = "declined" if r["declined"] else _fmt_score(r["eval_scores"])
        print(f"    → {r['latency_ms']}ms | {status}")
        results[q["id"]] = r
    return results


def run_comparison(model_a: str, model_b: str, ctx_a: int, ctx_b: int,
                   queries: list, index_path: str) -> dict:
    print("\n" + "=" * 70)
    print(f"  Model comparison: {model_a}  vs  {model_b}")
    print(f"  Index: {index_path}  |  Queries: {len(queries)}")
    print(f"  Strategy: two-pass (all A then all B) — avoids model-swap overhead")
    print("=" * 70)

    # Load pipeline once — both models share the same FAISS index + embeddings.
    print("\nLoading RAG pipeline (FAISS index + embeddings + reranker)…")
    from ragchatbot import CrossModalRAGChatbot
    chatbot = CrossModalRAGChatbot(index_path=index_path)

    # Run all queries with model A first, then all with model B.
    # On 8 GB Apple Silicon this is much faster: Ollama loads each model
    # once instead of evicting and reloading it on every query.
    res_a_map = _run_all_queries(chatbot, queries, model_a, ctx_a, "Model A")
    res_b_map = _run_all_queries(chatbot, queries, model_b, ctx_b, "Model B")

    results = [
        {"query": q, model_a: res_a_map[q["id"]], model_b: res_b_map[q["id"]]}
        for q in queries
    ]

    return {
        "run_at": datetime.now().isoformat(),
        "model_a": model_a,
        "model_b": model_b,
        "ctx_a": ctx_a,
        "ctx_b": ctx_b,
        "index_path": index_path,
        "results": results,
    }


def _fmt_score(eval_scores) -> str:
    if eval_scores is None:
        return "eval n/a"
    overall = eval_scores.get("overall_score", 0)
    verdict = eval_scores.get("verdict", "?")
    return f"overall={overall:.2f} ({verdict})"


def _avg(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def print_summary(report: dict):
    model_a = report["model_a"]
    model_b = report["model_b"]
    results = report["results"]

    def collect(model):
        latencies, scores, declined = [], [], 0
        for row in results:
            r = row[model]
            latencies.append(r["latency_ms"])
            if r["declined"]:
                declined += 1
            elif r["eval_scores"]:
                scores.append(r["eval_scores"].get("overall_score"))
        return latencies, scores, declined

    lat_a, scores_a, dec_a = collect(model_a)
    lat_b, scores_b, dec_b = collect(model_b)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {model_a:<22} {model_b:<22}")
    print("-" * 70)
    print(f"{'Avg latency':<30} {_avg(lat_a):>10}ms       {_avg(lat_b):>10}ms")
    print(f"{'Avg overall score':<30} {str(_avg(scores_a)):>10}        {str(_avg(scores_b)):>10}")
    print(f"{'Queries declined':<30} {dec_a:>10}          {dec_b:>10}")
    out_a = results[-2][model_a] if len(results) >= 2 else None
    out_b = results[-2][model_b] if len(results) >= 2 else None
    for row in results:
        if row["query"]["id"] == "out_of_domain":
            out_a = row[model_a]
            out_b = row[model_b]
            break
    if out_a and out_b:
        print(f"{'Out-of-domain declined?':<30} {'yes' if out_a['declined'] else 'NO (hallucination risk)':>10}  "
              f"{'yes' if out_b['declined'] else 'NO (hallucination risk)':>10}")
    print("=" * 70)


def save_json(report: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved → {path}")


def save_html(report: dict, path: str):
    model_a = report["model_a"]
    model_b = report["model_b"]
    rows_html = ""

    for row in report["results"]:
        q = row["query"]
        ra = row[model_a]
        rb = row[model_b]

        def cell(r):
            if r["declined"]:
                verdict_html = '<span style="color:#f59e0b">declined</span>'
                score_html = ""
            elif r["eval_scores"]:
                ov = r["eval_scores"].get("overall_score", 0)
                color = "#10b981" if ov >= 0.6 else "#ef4444"
                verdict_html = f'<span style="color:{color}">{r["eval_scores"].get("verdict","?")}</span>'
                score_html = (
                    f'faithfulness={r["eval_scores"].get("faithfulness",0):.2f} '
                    f'relevance={r["eval_scores"].get("answer_relevance",0):.2f} '
                    f'recall={r["eval_scores"].get("context_recall",0):.2f}'
                )
            else:
                verdict_html = "—"
                score_html = ""
            answer_preview = (r["answer"] or "")[:400].replace("<", "&lt;").replace(">", "&gt;")
            return (
                f"<td style='vertical-align:top;padding:8px;border:1px solid #333'>"
                f"<b>{r['latency_ms']}ms</b> {verdict_html}<br>"
                f"<small style='color:#9ca3af'>{score_html}</small><br>"
                f"<div style='font-size:12px;margin-top:4px'>{answer_preview}{'…' if len(r['answer'])>400 else ''}</div>"
                f"</td>"
            )

        rows_html += (
            f"<tr>"
            f"<td style='vertical-align:top;padding:8px;border:1px solid #333;min-width:200px'>"
            f"<b>{q['id']}</b><br>"
            f"<small style='color:#9ca3af'>{q['category']}</small><br>"
            f"<div style='font-size:12px;margin-top:4px'>{q['query']}</div>"
            f"</td>"
            + cell(ra)
            + cell(rb)
            + "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Model Comparison: {model_a} vs {model_b}</title>
  <style>
    body {{ background:#0f0f17; color:#e2e8f0; font-family:system-ui,sans-serif; padding:24px; }}
    h1 {{ font-size:1.4rem; margin-bottom:4px; }}
    .meta {{ color:#9ca3af; font-size:13px; margin-bottom:20px; }}
    table {{ border-collapse:collapse; width:100%; }}
    th {{ background:#1a1a2e; padding:10px 8px; border:1px solid #333; text-align:left; }}
  </style>
</head>
<body>
  <h1>Model Comparison</h1>
  <p class="meta">
    {model_a} (ctx={report['ctx_a']}) vs {model_b} (ctx={report['ctx_b']}) &nbsp;·&nbsp;
    {len(report['results'])} queries &nbsp;·&nbsp; {report['run_at']}
  </p>
  <table>
    <thead>
      <tr>
        <th style="width:22%">Query</th>
        <th style="width:39%">{model_a}</th>
        <th style="width:39%">{model_b}</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved → {path}")


# ── entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head model comparison on the legal RAG index"
    )
    parser.add_argument("--model-a", default="qwen2.5vl:3b",
                        help="First model (default: qwen2.5vl:3b)")
    parser.add_argument("--model-b", default="llava:7b",
                        help="Second model (default: llava:7b)")
    parser.add_argument("--ctx-a", type=int, default=2048,
                        help="num_ctx for model-a (default: 2048)")
    parser.add_argument("--ctx-b", type=int, default=1024,
                        help="num_ctx for model-b — keep ≤1024 on 8GB for 7B models (default: 1024)")
    parser.add_argument("--index", default=FAISS_INDEX_PATH,
                        help="Path to the FAISS index")
    parser.add_argument("--queries", default=None,
                        help="JSON file with custom queries (same schema as DEFAULT_QUERIES)")
    parser.add_argument("--out-json", default=RESULTS_JSON)
    parser.add_argument("--out-html", default=RESULTS_HTML)
    args = parser.parse_args()

    if not os.path.exists(args.index):
        print(f"ERROR: index not found at '{args.index}'")
        print("Upload files via the app first, or set --index to the correct path.")
        sys.exit(1)

    queries = DEFAULT_QUERIES
    if args.queries:
        with open(args.queries, encoding="utf-8") as f:
            queries = json.load(f)

    # Check Ollama is running and both models are available before starting
    try:
        import ollama as _ollama
        available = {m.model for m in _ollama.list().models}
        for model in (args.model_a, args.model_b):
            # Ollama tag matching: "llava:7b" matches "llava:7b" exactly or
            # as a prefix of "llava:7b-v1.6-mistral-q4_K_M"
            if not any(a.startswith(model.split(":")[0]) for a in available):
                print(f"ERROR: model '{model}' is not pulled.")
                print(f"  Run: ollama pull {model}")
                print("  Then re-run this script.")
                sys.exit(1)
    except Exception as e:
        print(f"ERROR: Ollama is not running or not reachable ({e})")
        print("  Start it with:  OLLAMA_FLASH_ATTENTION=1 ollama serve")
        print("  Then re-run this script.")
        sys.exit(1)

    report = run_comparison(
        model_a=args.model_a,
        model_b=args.model_b,
        ctx_a=args.ctx_a,
        ctx_b=args.ctx_b,
        queries=queries,
        index_path=args.index,
    )

    print_summary(report)
    save_json(report, args.out_json)
    save_html(report, args.out_html)

    print(f"\nOpen the report: open {args.out_html}")


if __name__ == "__main__":
    main()
