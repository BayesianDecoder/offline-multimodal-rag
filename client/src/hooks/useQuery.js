import { useMutation, useQueryClient } from '@tanstack/react-query';
import { api, getStoredSessionId, setStoredSessionId } from '../lib/api';

export function useSendQuery() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ query, modality }) => {
      const session_id = getStoredSessionId() || undefined;
      const response = await api.post('/api/query', { query, session_id, modality });
      return response.data;
    },
    onSuccess: (data) => {
      if (data.session_id) setStoredSessionId(data.session_id);
      queryClient.invalidateQueries({ queryKey: ['analytics-summary'] });
    }
  });
}
