import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
  PieChart, Pie, Cell, Legend, BarChart, Bar
} from 'recharts';
import { useAnalyticsSummary } from '../hooks/useAnalytics';

const MODALITY_COLORS = { pdf: '#7c3aed', docx: '#06b6d4', image: '#f59e0b', audio: '#10b981', text: '#64748b' };

function KpiCard({ label, value, accent }) {
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <p className="text-xs text-text-muted">{label}</p>
      <p className={`text-2xl font-semibold mt-1 ${accent || ''}`}>{value}</p>
    </div>
  );
}

export default function Analytics() {
  const { data, isLoading, isError } = useAnalyticsSummary();

  if (isLoading) return <div className="p-6 text-text-muted text-sm">Loading analytics…</div>;
  if (isError || !data) return <div className="p-6 text-red-400 text-sm">Failed to load analytics.</div>;

  const modalityData = Object.entries(data.queries_by_modality || {})
    .filter(([, count]) => count > 0)
    .map(([modality, count]) => ({ name: modality, value: count }));

  const hallucinationByModality = Object.entries(data.hallucination_rate_by_modality || {}).map(
    ([modality, rate]) => ({ modality, rate })
  );

  const latencyTrend = (data.latency_trend || []).map((point, i) => ({
    index: i + 1,
    latency_ms: point.latency_ms
  }));

  return (
    <div className="p-6 overflow-y-auto h-full">
      <h2 className="text-lg font-semibold mb-4">Analytics</h2>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <KpiCard label="Total Queries" value={data.total_queries} />
        <KpiCard label="Avg Latency" value={`${data.avg_latency_ms}ms`} />
        <KpiCard label="Cache Hit %" value={`${data.cache_hit_rate}%`} accent="text-secondary" />
        <KpiCard
          label="Hallucination Rate"
          value={`${data.hallucination_rate}%`}
          accent={data.hallucination_rate > 20 ? 'text-red-400' : 'text-emerald-400'}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        <div className="rounded-xl border border-border bg-surface p-4">
          <p className="text-sm font-medium mb-3">Latency — last {latencyTrend.length} queries</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={latencyTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#232330" />
              <XAxis dataKey="index" stroke="#9ca3af" fontSize={11} />
              <YAxis stroke="#9ca3af" fontSize={11} />
              <Tooltip contentStyle={{ background: '#111118', border: '1px solid #232330' }} />
              <ReferenceLine y={2000} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: '2s target', fill: '#f59e0b', fontSize: 10 }} />
              <Line type="monotone" dataKey="latency_ms" stroke="#7c3aed" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-xl border border-border bg-surface p-4">
          <p className="text-sm font-medium mb-3">Queries by modality</p>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={modalityData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80}>
                {modalityData.map((entry) => (
                  <Cell key={entry.name} fill={MODALITY_COLORS[entry.name] || '#64748b'} />
                ))}
              </Pie>
              <Legend />
              <Tooltip contentStyle={{ background: '#111118', border: '1px solid #232330' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="rounded-xl border border-border bg-surface p-4">
        <p className="text-sm font-medium mb-3">Hallucination rate by modality</p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={hallucinationByModality}>
            <CartesianGrid strokeDasharray="3 3" stroke="#232330" />
            <XAxis dataKey="modality" stroke="#9ca3af" fontSize={11} />
            <YAxis stroke="#9ca3af" fontSize={11} unit="%" />
            <Tooltip contentStyle={{ background: '#111118', border: '1px solid #232330' }} />
            <Bar dataKey="rate" fill="#f59e0b" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
