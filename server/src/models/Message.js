const mongoose = require('mongoose');

const messageSchema = new mongoose.Schema({
  session_id: { type: String, required: true, index: true },
  role: { type: String, enum: ['user', 'assistant'], required: true },
  content: { type: String, required: true },
  sources: { type: Array, default: [] },
  eval_scores: { type: Object, default: null },
  latency_ms: { type: Number, default: null },
  cache_hit: { type: Boolean, default: false },
  timestamp: { type: Date, default: Date.now, index: true }
});

module.exports = mongoose.model('Message', messageSchema);
