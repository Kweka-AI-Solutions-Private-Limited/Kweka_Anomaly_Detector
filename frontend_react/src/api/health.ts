import { apiClient } from './client';
import { HealthStatus } from '../types';

export async function getHealthStatus(): Promise<HealthStatus> {
  const response = await apiClient.get('/api/health');
  return response.data;
}
