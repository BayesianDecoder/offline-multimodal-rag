const express = require('express');
const Analytics = require('../models/Analytics');
const { logger } = require('../middleware/logger');

const router = express.Router();

router.get('/summary', async (req, res) => {
  try {
    const [totals] = await Analytics.aggregate([
      {
        $group: {
          _id: null,
          total_queries: { $sum: 1 },
          avg_latency_ms: { $avg: '$latency_ms' },
          cache_hits: { $sum: { $cond: ['$cache_hit', 1, 0] } },
          faithfulness_avg: { $avg: '$faithfulness' },
          scored_queries: { $sum: { $cond: [{ $ne: ['$overall_score', null] }, 1, 0] } },
          hallucinated_queries: {
            $sum: { $cond: [{ $and: [{ $ne: ['$overall_score', null] }, { $lt: ['$overall_score', 0.6] }] }, 1, 0] }
          }
        }
      }
    ]);

    const byModality = await Analytics.aggregate([
      { $group: { _id: '$modality', count: { $sum: 1 } } }
    ]);
    const queries_by_modality = { pdf: 0, image: 0, audio: 0, docx: 0, text: 0 };
    byModality.forEach((row) => {
      if (row._id in queries_by_modality) queries_by_modality[row._id] = row.count;
    });

    const hallucinationByModality = await Analytics.aggregate([
      { $match: { overall_score: { $ne: null } } },
      {
        $group: {
          _id: '$modality',
          scored: { $sum: 1 },
          hallucinated: { $sum: { $cond: [{ $lt: ['$overall_score', 0.6] }, 1, 0] } }
        }
      }
    ]);
    const hallucination_rate_by_modality = { pdf: 0, image: 0, audio: 0, docx: 0, text: 0 };
    hallucinationByModality.forEach((row) => {
      if (row._id in hallucination_rate_by_modality) {
        hallucination_rate_by_modality[row._id] = Math.round((row.hallucinated / row.scored) * 1000) / 10;
      }
    });

    const latencyTrend = await Analytics.find({}, { timestamp: 1, latency_ms: 1, _id: 0 })
      .sort({ timestamp: -1 })
      .limit(50)
      .lean();

    const total_queries = totals ? totals.total_queries : 0;
    const cache_hit_rate = total_queries > 0 ? Math.round((totals.cache_hits / total_queries) * 1000) / 10 : 0;
    const hallucination_rate = totals && totals.scored_queries > 0
      ? Math.round((totals.hallucinated_queries / totals.scored_queries) * 1000) / 10
      : 0;

    return res.json({
      success: true,
      total_queries,
      avg_latency_ms: totals ? Math.round(totals.avg_latency_ms || 0) : 0,
      cache_hit_rate,
      hallucination_rate,
      queries_by_modality,
      hallucination_rate_by_modality,
      latency_trend: latencyTrend.reverse(),
      faithfulness_avg: totals ? Math.round((totals.faithfulness_avg || 0) * 100) / 100 : 0
    });
  } catch (error) {
    logger.error('analytics summary failed', { error: error.message });
    return res.status(500).json({ success: false, error: error.message });
  }
});

module.exports = router;
