"""
hallucination_eval.py - Lightweight, no-extra-model hallucination/quality
evaluation for RAG answers.

Reuses the pipeline's existing components:
  - `cross_encoder`: the same sentence_transformers.CrossEncoder used for
    reranking (e.g. chatbot_instance.search_engine.reranker)
  - `encoder`: the same CLIP embedding generator used for retrieval
    (e.g. chatbot_instance.generator, exposing .embed(text) -> np.ndarray)
"""

import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(answer: str) -> list:
    """Split `answer` into non-empty sentences on ./!/? boundaries."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer.strip()) if s.strip()]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# ============================================================================
# METRIC 1 — FAITHFULNESS
# ============================================================================

def check_faithfulness(answer: str, chunks: list, cross_encoder, threshold: float = 0.3) -> dict:
    """Score how well each sentence of `answer` is grounded in `chunks`.

    Each sentence is paired with every chunk and scored with the existing
    cross-encoder reranker; the cross-encoder's raw logit is squashed
    through a sigmoid into a 0-1 pseudo-probability before thresholding,
    since ms-marco style cross-encoders output unbounded logits, not
    probabilities.

    Returns:
        {
          "faithfulness_score": float 0-1,
          "hallucinated_sentences": [{"sentence": str, "score": float}, ...],
          "sentence_breakdown": [{"sentence": str, "score": float, "grounded": bool}, ...]
        }
    """
    sentences = _split_sentences(answer)
    if not sentences or not chunks:
        return {
            "faithfulness_score": 0.0,
            "hallucinated_sentences": [],
            "sentence_breakdown": [],
        }

    pairs = [[sentence, chunk] for sentence in sentences for chunk in chunks]
    raw_scores = np.asarray(cross_encoder.predict(pairs), dtype=np.float64)
    scores = _sigmoid(raw_scores).reshape(len(sentences), len(chunks))
    best_scores = scores.max(axis=1)

    sentence_breakdown = []
    hallucinated_sentences = []
    grounded_count = 0
    for sentence, score in zip(sentences, best_scores):
        grounded = bool(score >= threshold)
        if grounded:
            grounded_count += 1
        else:
            hallucinated_sentences.append({"sentence": sentence, "score": float(score)})
        sentence_breakdown.append({"sentence": sentence, "score": float(score), "grounded": grounded})

    return {
        "faithfulness_score": grounded_count / len(sentences),
        "hallucinated_sentences": hallucinated_sentences,
        "sentence_breakdown": sentence_breakdown,
    }


# ============================================================================
# RETRIEVAL CONFIDENCE — gate generation on whether anything relevant
# was actually found, before spending an LLM call on it
# ============================================================================

def check_retrieval_confidence(query: str, chunks: list, cross_encoder, threshold: float = 0.05) -> dict:
    """Score `query` directly against each retrieved chunk with the same
    cross-encoder (this is the reranker's native use case — query-vs-passage
    relevance — unlike check_faithfulness's sentence-vs-passage repurposing).

    Meant to run right after retrieval, before generation: if nothing
    retrieved clears the threshold, the corpus has no relevant information
    for this query and generation should be skipped rather than letting the
    LLM answer from its own parametric knowledge.

    Threshold note: the ms-marco MiniLM cross-encoder outputs unbounded
    logits that are squashed through sigmoid here. On legal documents the
    L-2 model tends to produce negative logits (~-2 to -3) for in-corpus
    queries, which map to sigmoid scores of 0.04–0.18. True out-of-domain
    queries land at < 0.001. A threshold of 0.05 reliably separates them.

    Returns:
        {"confident": bool, "best_score": float, "scores": [float, ...]}
    """
    if not chunks:
        return {"confident": False, "best_score": 0.0, "scores": []}

    pairs = [[query, chunk] for chunk in chunks]
    raw_scores = np.asarray(cross_encoder.predict(pairs), dtype=np.float64)
    scores = _sigmoid(raw_scores)
    best_score = float(scores.max())

    return {
        "confident": best_score >= threshold,
        "best_score": best_score,
        "scores": [float(s) for s in scores],
    }


# ============================================================================
# METRIC 2 — ANSWER RELEVANCE
# ============================================================================

def check_answer_relevance(query: str, answer: str, encoder) -> float:
    """Cosine similarity between the query and answer, embedded with the
    existing CLIP encoder. Returns a float in [0, 1]."""
    query_emb = np.asarray(encoder.embed(query)).reshape(1, -1)
    answer_emb = np.asarray(encoder.embed(answer)).reshape(1, -1)
    similarity = cosine_similarity(query_emb, answer_emb)[0][0]
    return float(max(0.0, min(1.0, similarity)))


# ============================================================================
# METRIC 3 — CONTEXT RECALL
# ============================================================================

def check_context_recall(answer: str, chunks: list) -> float:
    """Fraction of retrieved `chunks` whose first 10 words show up (>=2 of
    them) in `answer`, as a cheap proxy for "was this chunk actually used"."""
    if not chunks:
        return 0.0

    answer_lower = answer.lower()
    used_chunks = 0
    for chunk in chunks:
        first_words = chunk.lower().split()[:10]
        hits = sum(1 for word in first_words if word in answer_lower)
        if hits >= 2:
            used_chunks += 1

    return used_chunks / len(chunks)


# ============================================================================
# MASTER FUNCTION
# ============================================================================

def evaluate_response(query: str, answer: str, chunks: list, encoder, cross_encoder) -> dict:
    """Run all three metrics and combine them into an overall verdict.

    Returns:
        {
          "faithfulness": float,
          "answer_relevance": float,
          "context_recall": float,
          "overall_score": float,
          "verdict": "✅ Reliable" | "⚠️ May Hallucinate",
          "hallucinated_sentences": list
        }
    """
    faithfulness_result = check_faithfulness(answer, chunks, cross_encoder)
    answer_relevance = check_answer_relevance(query, answer, encoder)
    context_recall = check_context_recall(answer, chunks)

    faithfulness = faithfulness_result["faithfulness_score"]
    overall_score = (faithfulness + answer_relevance + context_recall) / 3
    verdict = "✅ Reliable" if overall_score > 0.6 else "⚠️ May Hallucinate"

    return {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_recall": context_recall,
        "overall_score": overall_score,
        "verdict": verdict,
        "hallucinated_sentences": faithfulness_result["hallucinated_sentences"],
    }


# ============================================================================
# HTML PANEL (for the existing Flask/Jinja UI)
# ============================================================================

def _score_emoji(score: float) -> str:
    if score > 0.6:
        return "🟢"
    if score >= 0.4:
        return "🟡"
    return "🔴"


def render_eval_panel(eval_scores: dict) -> str:
    """Render `eval_scores` (as returned by evaluate_response) into an HTML
    snippet for the response-quality panel."""
    faithfulness = eval_scores.get("faithfulness", 0.0)
    answer_relevance = eval_scores.get("answer_relevance", 0.0)
    context_recall = eval_scores.get("context_recall", 0.0)
    overall_score = eval_scores.get("overall_score", 0.0)
    verdict = eval_scores.get("verdict", "")
    hallucinated_sentences = eval_scores.get("hallucinated_sentences", [])

    hallucinated_html = ""
    if hallucinated_sentences:
        items = "".join(
            f"<li><s>{item['sentence']}</s> (score: {item['score']:.2f})</li>"
            for item in hallucinated_sentences
        )
        hallucinated_html = f"<div class='eval-hallucinations'><strong>Unsupported statements:</strong><ul>{items}</ul></div>"

    return f"""<div class="eval-panel">
  <div>Faithfulness: {faithfulness:.2f} {_score_emoji(faithfulness)}</div>
  <div>Answer Relevance: {answer_relevance:.2f} {_score_emoji(answer_relevance)}</div>
  <div>Context Recall: {context_recall:.2f} {_score_emoji(context_recall)}</div>
  <div><strong>Overall: {overall_score:.2f} — {verdict}</strong></div>
  {hallucinated_html}
</div>"""
