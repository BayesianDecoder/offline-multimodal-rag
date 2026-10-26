const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const Document = require('../models/Document');
const { logger } = require('../middleware/logger');

const router = express.Router();
const FLASK_URL = process.env.FLASK_URL || 'http://localhost:5001';

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 500 * 1024 * 1024 } });

const EXT_TO_TYPE = {
  pdf: 'pdf', docx: 'docx', doc: 'docx',
  png: 'image', jpg: 'image', jpeg: 'image', gif: 'image', bmp: 'image', tiff: 'image', webp: 'image',
  mp3: 'audio', wav: 'audio', m4a: 'audio', flac: 'audio', ogg: 'audio', opus: 'audio', wma: 'audio', aac: 'audio'
};

function inferType(filename) {
  const ext = (filename.split('.').pop() || '').toLowerCase();
  return EXT_TO_TYPE[ext] || 'pdf';
}

router.get('/', async (req, res) => {
  try {
    const documents = await Document.find().sort({ indexed_at: -1 }).lean();
    return res.json({ success: true, count: documents.length, documents });
  } catch (error) {
    logger.error('list documents failed', { error: error.message });
    return res.status(500).json({ success: false, error: error.message });
  }
});

router.post('/', upload.array('files'), async (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({ success: false, error: 'No files provided' });
  }

  try {
    const form = new FormData();
    req.files.forEach((file) => {
      form.append('files', file.buffer, { filename: file.originalname });
    });

    const flaskResponse = await axios.post(`${FLASK_URL}/upload`, form, {
      headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 300000
    });

    const data = flaskResponse.data;
    if (!data.success) {
      return res.status(502).json({ success: false, error: data.error || 'Flask upload failed' });
    }

    const docs = await Document.insertMany(
      req.files.map((file) => ({
        filename: file.originalname,
        type: inferType(file.originalname),
        size_bytes: file.size,
        status: 'indexed'
      }))
    );

    return res.json({ success: true, ...data, documents: docs });
  } catch (error) {
    logger.error('upload proxy failed', { error: error.message });
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({ success: false, error: 'RAG service (Flask) is unreachable' });
    }
    return res.status(500).json({ success: false, error: error.message });
  }
});

module.exports = router;
