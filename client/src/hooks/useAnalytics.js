import { useQuery } from '@tanstack/react-query';
import { api } from '../lib/api';

export function useAnalyticsSummary() {
  return useQuery({
    queryKey: ['analytics-summary'],
    queryFn: async () => {
      const response = await api.get('/api/analytics/summary');
      return response.data;
    },
    refetchInterval: 30000
  });
}
