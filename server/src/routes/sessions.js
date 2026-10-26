const express = require('express');
const axios = require('axios');
const Message = require('../models/Message');
const { logger } = require('../middleware/logger');

const router = express.Router();
const FLASK_URL = process.env.FLASK_URL || 'http://localhost:5001';

router.get('/', async (req, res) => {
  try {
    const sessions = await Message.aggregate([
      {
        $group: {
          _id: '$session_id',
          turn_count: { $sum: 1 },
          last_activity: { $max: '$timestamp' },
          first_message: { $first: '$content' }
        }
      },
      { $sort: { last_activity: -1 } },
      { $limit: 100 }
    ]);

    return res.json({
      success: true,
      sessions: sessions.map((s) => ({
        session_id: s._id,
        turn_count: s.turn_count,
        last_activity: s.last_activity
      }))
    });
  } catch (error) {
    logger.error('list sessions failed', { error: error.message });
    return res.status(500).json({ success: false, error: error.message });
  }
});

router.delete('/:id', async (req, res) => {
  const sessionId = req.params.id;
  try {
    await axios.post(`${FLASK_URL}/session/clear`, { session_id: sessionId }, { timeout: 5000 });
  } catch (error) {
    logger.warn('flask session clear failed, continuing with local delete', { error: error.message });
  }

  try {
    const result = await Message.deleteMany({ session_id: sessionId });
    return res.json({ success: true, deleted_messages: result.deletedCount });
  } catch (error) {
    logger.error('delete session failed', { error: error.message });
    return res.status(500).json({ success: false, error: error.message });
  }
});

module.exports = router;
