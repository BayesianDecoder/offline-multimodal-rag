require('dotenv').config();

const express = require('express');
const http = require('http');
const cors = require('cors');
const mongoose = require('mongoose');
const { Server } = require('socket.io');

const { logger, requestLogger } = require('./middleware/logger');
const { apiRateLimiter } = require('./middleware/rateLimiter');

const queryRoutes = require('./routes/query');
const documentRoutes = require('./routes/documents');
const sessionRoutes = require('./routes/sessions');
const analyticsRoutes = require('./routes/analytics');
const cacheRoutes = require('./routes/cache');

const PORT = process.env.PORT || 5000;
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/legal_rag';
const CLIENT_ORIGIN = process.env.CLIENT_ORIGIN || 'http://localhost:3000';

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: CLIENT_ORIGIN } });

app.use(cors({ origin: CLIENT_ORIGIN }));
app.use(express.json({ limit: '50mb' }));
app.use(requestLogger);

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    mongo: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected'
  });
});

app.use('/api/query', apiRateLimiter, queryRoutes);
app.use('/api/documents', apiRateLimiter, documentRoutes);
app.use('/api/sessions', apiRateLimiter, sessionRoutes);
app.use('/api/analytics', apiRateLimiter, analyticsRoutes);
app.use('/api/cache', apiRateLimiter, cacheRoutes);

app.use((req, res) => {
  res.status(404).json({ success: false, error: 'Endpoint not found', path: req.originalUrl });
});

// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  logger.error('unhandled error', { error: err.message, stack: err.stack });
  res.status(500).json({ success: false, error: 'Internal server error' });
});

io.on('connection', (socket) => {
  logger.info('socket connected', { id: socket.id });
  socket.on('disconnect', () => logger.info('socket disconnected', { id: socket.id }));
});

async function connectMongoWithRetry(retries = 10, delayMs = 3000) {
  for (let attempt = 1; attempt <= retries; attempt += 1) {
    try {
      await mongoose.connect(MONGO_URI);
      logger.info('MongoDB connected');
      return;
    } catch (error) {
      logger.warn(`MongoDB connection attempt ${attempt}/${retries} failed: ${error.message}`);
      if (attempt === retries) {
        logger.error('MongoDB unreachable after all retries — starting server anyway');
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
}

connectMongoWithRetry().then(() => {
  server.listen(PORT, () => {
    console.log('='.repeat(70));
    console.log(`Legal RAG API gateway listening on :${PORT}`);
    console.log(`Flask RAG service: ${process.env.FLASK_URL || 'http://localhost:5001'}`);
    console.log(`Mongo: ${MONGO_URI}`);
    console.log('='.repeat(70));
  });
});

module.exports = { app, server, io };
