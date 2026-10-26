const mongoose = require('mongoose');

const documentSchema = new mongoose.Schema({
  filename: { type: String, required: true },
  type: { type: String, enum: ['pdf', 'docx', 'image', 'audio'], required: true },
  size_bytes: { type: Number, required: true },
  indexed_at: { type: Date, default: Date.now },
  chunk_count: { type: Number, default: 0 },
  status: { type: String, enum: ['pending', 'processing', 'indexed', 'failed'], default: 'pending' }
});

module.exports = mongoose.model('Document', documentSchema);
