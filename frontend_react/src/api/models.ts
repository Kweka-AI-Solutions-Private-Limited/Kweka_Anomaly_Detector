import { apiClient } from './client';
import { Model, ModelVersion, ReferenceImage } from '../types';

export async function getModels(): Promise<Model[]> {
  const response = await apiClient.get('/api/models');
  return response.data;
}

export async function getModel(modelId: string): Promise<Model> {
  const response = await apiClient.get(`/api/models/${modelId}`);
  return response.data;
}

export async function createModel(
  name: string,
  description?: string,
  domain?: string,
  group_id?: string | null
): Promise<Model> {
  const response = await apiClient.post(
    '/api/models',
    { name, description, domain, group_id }
  );
  return response.data;
}

export async function updateModel(
  modelId: string,
  data: {
    name?: string;
    description?: string;
    domain?: string;
    group_id?: string | null;
  }
): Promise<Model> {
  const response = await apiClient.patch(`/api/models/${modelId}`, data);
  return response.data;
}

export async function getReferenceImages(modelId: string): Promise<ReferenceImage[]> {
  const response = await apiClient.get(`/api/models/${modelId}/references`);
  return response.data;
}

export async function uploadReferenceImages(modelId: string, files: File[]): Promise<ReferenceImage[]> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  const response = await apiClient.post(
    `/api/models/${modelId}/references`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
}

export async function buildModelVersion(modelId: string): Promise<ModelVersion> {
  const response = await apiClient.post(
    `/api/models/${modelId}/build`,
    {}
  );
  return response.data;
}

export async function getModelVersions(modelId: string): Promise<ModelVersion[]> {
  const response = await apiClient.get(`/api/models/${modelId}/versions`);
  return response.data;
}

export async function activateModel(modelId: string): Promise<Model> {
  const response = await apiClient.post(
    `/api/models/${modelId}/activate`,
    {}
  );
  return response.data;
}

export async function deactivateModel(modelId: string): Promise<Model> {
  const response = await apiClient.post(
    `/api/models/${modelId}/deactivate`,
    {}
  );
  return response.data;
}

export async function getVersionReferenceImages(modelId: string, versionId: string): Promise<ReferenceImage[]> {
  const response = await apiClient.get(`/api/models/${modelId}/versions/${versionId}/references`);
  return response.data;
}

export async function updateModelThreshold(modelId: string, threshold: number): Promise<ModelVersion> {
  const response = await apiClient.post(
    `/api/models/${modelId}/threshold`,
    { threshold }
  );
  return response.data;
}

export async function activateModelVersion(modelId: string, versionId: string): Promise<Model> {
  const response = await apiClient.post(
    `/api/models/${modelId}/versions/${versionId}/activate`,
    {}
  );
  return response.data;
}

export async function deleteModelVersion(modelId: string, versionId: string): Promise<{ message: string; version_id: string; model_id: string }> {
  const response = await apiClient.delete(`/api/models/${modelId}/versions/${versionId}`);
  return response.data;
}

export async function deleteModel(modelId: string): Promise<{ message: string; model_id: string }> {
  const response = await apiClient.delete(`/api/models/${modelId}`);
  return response.data;
}



