import { apiClient } from './client';
import { Inspection, InspectionResultResponse } from '../types';

export async function runInspection(
  modelId: string,
  file: File,
  thresholdOverride?: number,
  inspectionMode: 'single_image' | 'multi_instance' = 'single_image',
  minInstanceArea?: number,
  maxInstances?: number
): Promise<InspectionResultResponse> {
  const formData = new FormData();
  formData.append('model_id', modelId);
  if (thresholdOverride !== undefined && thresholdOverride !== null) {
    formData.append('threshold_override', thresholdOverride.toString());
  }
  formData.append('inspection_mode', inspectionMode);
  if (minInstanceArea !== undefined) {
    formData.append('min_instance_area', minInstanceArea.toString());
  }
  if (maxInstances !== undefined) {
    formData.append('max_instances', maxInstances.toString());
  }
  formData.append('image', file);

  const response = await apiClient.post('/api/inspections', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

export async function getInspections(params?: {
  model_id?: string;
  model_version_id?: string;
  run_id?: string;
  status?: string;
  defect_type?: string;
  location?: string;
  severity?: string;
  start_date?: string;
  end_date?: string;
  search?: string;
  skip?: number;
  limit?: number;
}): Promise<Inspection[]> {
  const response = await apiClient.get('/api/inspections', {
    params,
  });
  return response.data;
}

export async function getInspection(inspectionId: string): Promise<InspectionResultResponse> {
  const response = await apiClient.get(`/api/inspections/${inspectionId}`);
  return response.data;
}

export async function retryVlmAnalysis(inspectionId: string, force?: boolean): Promise<InspectionResultResponse> {
  const response = await apiClient.post(`/api/inspections/${inspectionId}/analyze-vlm`, {}, {
    params: force ? { force: true } : undefined,
  });
  return response.data;
}

export async function retryInstanceVlmAnalysis(inspectionId: string, instanceId: number, force?: boolean): Promise<InspectionResultResponse> {
  const response = await apiClient.post(`/api/inspections/${inspectionId}/instances/${instanceId}/analyze-vlm`, {}, {
    params: force ? { force: true } : undefined,
  });
  return response.data;
}


