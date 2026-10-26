const rateLimit = require('express-rate-limit');

const apiRateLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 30,
  standardHeaders: true,
  legacyHeaders: false,
  message: { success: false, error: 'Too many requests. Please slow down and try again shortly.' }
});

module.exports = { apiRateLimiter };
