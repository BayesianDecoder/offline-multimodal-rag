const mongoose = require('mongoose');

const analyticsSchema = new mongoose.Schema({
  timestamp: { type: Date, default: Date.now, index: true },
  latency_ms: { type: Number, required: true },
  cache_hit: { type: Boolean, default: false },
  modality: { type: String, enum: ['pdf', 'docx', 'image', 'audio', 'text'], default: 'text' },
  faithfulness: { type: Number, default: null },
  overall_score: { type: Number, default: null }
});

module.exports = mongoose.model('Analytics', analyticsSchema);
