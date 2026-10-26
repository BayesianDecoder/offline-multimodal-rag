const express = require('express');
const axios = require('axios');
const Message = require('../models/Message');
const Analytics = require('../models/Analytics');
const { validateQuery } = require('../middleware/validator');
const { logger } = require('../middleware/logger');

const router = express.Router();
const FLASK_URL = process.env.FLASK_URL || 'http://localhost:5001';

function detectModality(sources) {
  if (!sources || sources.length === 0) return 'text';
  const source = (sources[0].source || '').toLowerCase();
  if (source.endsWith('.pdf')) return 'pdf';
  if (source.endsWith('.docx') || source.endsWith('.doc')) return 'docx';
  if (/\.(png|jpg|jpeg|gif|bmp|tiff|webp)$/.test(source)) return 'image';
  if (/\.(mp3|wav|m4a|flac|ogg|opus|wma|aac)$/.test(source)) return 'audio';
  return 'text';
}

router.post('/', validateQuery, async (req, res) => {
  const { query, session_id, debug } = req.body;
  const start = Date.now();

  try {
    const flaskResponse = await axios.post(
      `${FLASK_URL}/query`,
      { query, session_id, debug: debug || false },
      { timeout: 120000 }  // 2 min: first query loads the model into Ollama
    );

    const latency_ms = Date.now() - start;
    const data = flaskResponse.data;

    if (!data.success) {
      return res.status(502).json({ success: false, error: data.error || 'Flask query failed' });
    }

    const modality = detectModality(data.sources);

    // Persist conversation turns + analytics; a logging failure must not
    // block the response the user is waiting on.
    try {
      await Message.insertMany([
        { session_id: data.session_id, role: 'user', content: query },
        {
          session_id: data.session_id,
          role: 'assistant',
          content: data.answer,
          sources: data.sources || [],
          eval_scores: data.eval_scores || null,
          latency_ms,
          cache_hit: !!data.cache_hit
        }
      ]);

      await Analytics.create({
        latency_ms,
        cache_hit: !!data.cache_hit,
        modality,
        faithfulness: data.eval_scores ? data.eval_scores.faithfulness : null,
        overall_score: data.eval_scores ? data.eval_scores.overall_score : null
      });
    } catch (persistError) {
      logger.warn('failed to persist message/analytics', { error: persistError.message });
    }

    return res.json({ ...data, latency_ms });
  } catch (error) {
    logger.error('query proxy failed', { error: error.message });
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({ success: false, error: 'RAG service (Flask) is unreachable' });
    }
    if (error.code === 'ECONNABORTED') {
      return res.status(504).json({ success: false, error: 'RAG service timed out' });
    }
    return res.status(500).json({ success: false, error: error.message });
  }
});

router.get('/history', async (req, res) => {
  try {
    const { session_id, limit = 50, from, to } = req.query;
    const filter = {};
    if (session_id) filter.session_id = session_id;
    if (from || to) {
      filter.timestamp = {};
      if (from) filter.timestamp.$gte = new Date(from);
      if (to) filter.timestamp.$lte = new Date(to);
    }

    const messages = await Message.find(filter)
      .sort({ timestamp: -1 })
      .limit(Math.min(parseInt(limit, 10) || 50, 200))
      .lean();

    return res.json({ success: true, count: messages.length, messages: messages.reverse() });
  } catch (error) {
    logger.error('history fetch failed', { error: error.message });
    return res.status(500).json({ success: false, error: error.message });
  }
});

module.exports = router;
