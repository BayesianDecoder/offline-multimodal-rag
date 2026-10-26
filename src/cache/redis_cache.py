"""
redis_cache.py - Three-layer Redis caching for the RAG pipeline.

Layer 1: embedding cache      (emb:{md5})     TTL 1h
Layer 2: retrieval cache      (ret:{md5})      TTL 30m
Layer 3: session conversation memory (session:{uuid}) TTL 1h, refreshed on write

Every Redis call is wrapped so a down/unreachable Redis degrades the
system to "no cache" instead of crashing the request.
"""

import hashlib
import json
import logging
import uuid

import numpy as np
import redis

logger = logging.getLogger(__name__)

EMBEDDING_TTL_SECONDS = 3600
RETRIEVAL_TTL_SECONDS = 1800
SESSION_TTL_SECONDS = 3600
SESSION_MAX_TURNS = 6


class RAGCache:
    """Redis-backed cache for embeddings, retrieval results, and session history."""

    def __init__(self, host: str = "localhost", port: int = 6379):
        """Connect to Redis. Connectivity is not required at construction time —
        every method below tolerates a down Redis and falls back to cache-miss
        behaviour."""
        self.r = redis.Redis(host=host, port=port, decode_responses=False)
        try:
            self.r.ping()
            logger.debug("Redis connection OK (%s:%s)", host, port)
        except redis.RedisError as e:
            logger.warning("Redis unavailable at startup (%s) — caching disabled until it recovers", e)

    @staticmethod
    def _normalize(text: str) -> str:
        return text.lower().strip()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # LAYER 1: embedding cache
    # ------------------------------------------------------------------

    def get_embedding(self, text: str):
        """Return a cached embedding (np.ndarray) for `text`, or None on miss/error."""
        key = f"emb:{self._hash(self._normalize(text))}"
        try:
            raw = self.r.get(key)
        except redis.RedisError as e:
            logger.debug("get_embedding: Redis error (%s) — treating as miss", e)
            return None
        if raw is None:
            logger.debug("get_embedding: MISS %s", key)
            return None
        logger.debug("get_embedding: HIT %s", key)
        return np.frombuffer(raw, dtype=np.float32)

    def set_embedding(self, text: str, embedding: np.ndarray):
        """Cache `embedding` for `text` with a 1 hour TTL. No-op if Redis is
        down or `embedding` can't be converted to bytes."""
        key = f"emb:{self._hash(self._normalize(text))}"
        try:
            self.r.setex(key, EMBEDDING_TTL_SECONDS, np.asarray(embedding, dtype=np.float32).tobytes())
            logger.debug("set_embedding: stored %s", key)
        except (redis.RedisError, TypeError, ValueError) as e:
            logger.debug("set_embedding: error (%s) — skipping cache write", e)

    # ------------------------------------------------------------------
    # LAYER 2: retrieval cache
    # ------------------------------------------------------------------

    def get_retrieval(self, query: str):
        """Return cached retrieved chunks (list) for `query`, or None on miss/error."""
        key = f"ret:{self._hash(self._normalize(query))}"
        try:
            raw = self.r.get(key)
        except redis.RedisError as e:
            logger.debug("get_retrieval: Redis error (%s) — treating as miss", e)
            return None
        if raw is None:
            logger.debug("get_retrieval: MISS %s", key)
            return None
        logger.debug("get_retrieval: HIT %s", key)
        return json.loads(raw)

    def set_retrieval(self, query: str, chunks: list):
        """Cache retrieved `chunks` for `query` with a 30 minute TTL. No-op if
        Redis is down or `chunks` contains something non-JSON-serializable
        (e.g. an unexpected type from the ingestion pipeline's metadata)."""
        key = f"ret:{self._hash(self._normalize(query))}"
        try:
            self.r.setex(key, RETRIEVAL_TTL_SECONDS, json.dumps(chunks))
            logger.debug("set_retrieval: stored %s", key)
        except (redis.RedisError, TypeError, ValueError) as e:
            logger.debug("set_retrieval: error (%s) — skipping cache write", e)

    # ------------------------------------------------------------------
    # LAYER 3: session conversation memory
    # ------------------------------------------------------------------

    def create_session(self) -> str:
        """Create a new session id and initialize empty history for it."""
        session_id = str(uuid.uuid4())
        key = f"session:{session_id}"
        try:
            self.r.setex(key, SESSION_TTL_SECONDS, json.dumps([]))
            logger.debug("create_session: %s", session_id)
        except redis.RedisError as e:
            logger.debug("create_session: Redis error (%s) — session will not persist", e)
        return session_id

    def get_history(self, session_id: str) -> list:
        """Return the turn history for `session_id`, or [] if missing/error."""
        key = f"session:{session_id}"
        try:
            raw = self.r.get(key)
        except redis.RedisError as e:
            logger.debug("get_history: Redis error (%s) — returning empty history", e)
            return []
        if raw is None:
            return []
        return json.loads(raw)

    def add_turn(self, session_id: str, role: str, content: str):
        """Append a turn to the session, keep only the last SESSION_MAX_TURNS,
        and reset the TTL to 1 hour."""
        key = f"session:{session_id}"
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        history = history[-SESSION_MAX_TURNS:]
        try:
            self.r.setex(key, SESSION_TTL_SECONDS, json.dumps(history))
            logger.debug("add_turn: %s now has %d turns", session_id, len(history))
        except redis.RedisError as e:
            logger.debug("add_turn: Redis error (%s) — turn not persisted", e)

    def build_prompt_with_history(self, query: str, context: str, session_id: str) -> str:
        """Build a prompt string combining prior conversation turns, retrieved
        context, and the current question."""
        history = self.get_history(session_id)
        history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
        return (
            f"Previous conversation:\n{history_text}\n"
            f"Context:\n{context}\n"
            f"Current question: {query}\nAnswer:"
        )

    def invalidate_document_caches(self):
        """Drop all embedding and retrieval cache entries. Call this whenever
        the underlying document index changes (new upload / clear) — otherwise
        a cached retrieval could keep answering from a since-replaced index."""
        try:
            keys = self.r.keys("emb:*") + self.r.keys("ret:*")
            if keys:
                self.r.delete(*keys)
            logger.debug("invalidate_document_caches: dropped %d keys", len(keys))
        except redis.RedisError as e:
            logger.debug("invalidate_document_caches: Redis error (%s) — nothing to invalidate", e)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_cache_stats(self) -> dict:
        """Return counts of keys per cache layer plus Redis memory usage in MB."""
        try:
            embedding_keys = len(self.r.keys("emb:*"))
            retrieval_keys = len(self.r.keys("ret:*"))
            session_keys = len(self.r.keys("session:*"))
            info = self.r.info("memory")
            memory_used_mb = round(info.get("used_memory", 0) / 1e6, 2)
        except redis.RedisError as e:
            logger.debug("get_cache_stats: Redis error (%s) — returning zeroed stats", e)
            return {
                "embedding_keys": 0,
                "retrieval_keys": 0,
                "session_keys": 0,
                "memory_used_mb": 0,
                "redis_available": False,
            }
        return {
            "embedding_keys": embedding_keys,
            "retrieval_keys": retrieval_keys,
            "session_keys": session_keys,
            "memory_used_mb": memory_used_mb,
            "redis_available": True,
        }
