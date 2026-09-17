import { apiClient } from './client';
import { InspectionRun } from '../types';

export async function getInspectionRuns(params?: {
  model_id?: string;
  model_version_id?: string;
  status?: string;
}): Promise<InspectionRun[]> {
  const response = await apiClient.get('/api/inspection-runs', {
    params,
  });
  return response.data;
}

export async function getInspectionRun(runId: string): Promise<InspectionRun> {
  const response = await apiClient.get(`/api/inspection-runs/${runId}`);
  return response.data;
}

export async function createInspectionRun(
  modelId: string,
  files: File[],
  thresholdOverride?: number,
  inspectionMode?: string
): Promise<InspectionRun> {
  const formData = new FormData();
  formData.append('model_id', modelId);
  if (thresholdOverride !== undefined && thresholdOverride !== null) {
    formData.append('threshold_override', thresholdOverride.toString());
  }
  if (inspectionMode) {
    formData.append('inspection_mode', inspectionMode);
  }
  files.forEach((file) => {
    formData.append('files', file);
  });

  const response = await apiClient.post('/api/inspection-runs', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

