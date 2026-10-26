const Joi = require('joi');

const querySchema = Joi.object({
  query: Joi.string().min(3).max(1000).required(),
  session_id: Joi.string().uuid().optional(),
  modality: Joi.string().valid('text', 'image', 'audio').optional(),
  debug: Joi.boolean().optional()
});

function validateQuery(req, res, next) {
  const { error, value } = querySchema.validate(req.body, { abortEarly: false });
  if (error) {
    return res.status(400).json({
      success: false,
      error: 'Validation failed',
      details: error.details.map((d) => d.message)
    });
  }
  req.body = value;
  next();
}

module.exports = { validateQuery };
