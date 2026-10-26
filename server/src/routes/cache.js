const express = require('express');
const axios = require('axios');
const { logger } = require('../middleware/logger');

const router = express.Router();
const FLASK_URL = process.env.FLASK_URL || 'http://localhost:5001';

router.get('/stats', async (req, res) => {
  try {
    const response = await axios.get(`${FLASK_URL}/cache/stats`, { timeout: 5000 });
    return res.json(response.data);
  } catch (error) {
    logger.error('cache stats proxy failed', { error: error.message });
    return res.status(502).json({ error: 'Could not reach RAG service cache stats' });
  }
});

module.exports = router;
