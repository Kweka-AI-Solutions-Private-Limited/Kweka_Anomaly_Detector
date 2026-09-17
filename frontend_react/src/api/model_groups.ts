import { apiClient } from './client';
import { ModelGroup } from '../types';

export const getModelGroups = async (): Promise<ModelGroup[]> => {
  const response = await apiClient.get<ModelGroup[]>('/api/model-groups');
  return response.data;
};

export const createModelGroup = async (data: {
  name: string;
  description?: string;
}): Promise<ModelGroup> => {
  const response = await apiClient.post<ModelGroup>('/api/model-groups', data);
  return response.data;
};

export const getModelGroup = async (groupId: string): Promise<ModelGroup> => {
  const response = await apiClient.get<ModelGroup>(`/api/model-groups/${groupId}`);
  return response.data;
};

export const updateModelGroup = async (
  groupId: string,
  data: { name?: string; description?: string }
): Promise<ModelGroup> => {
  const response = await apiClient.patch<ModelGroup>(`/api/model-groups/${groupId}`, data);
  return response.data;
};

export const deleteModelGroup = async (
  groupId: string
): Promise<{ message: string; group_id: string }> => {
  const response = await apiClient.delete<{ message: string; group_id: string }>(`/api/model-groups/${groupId}`);
  return response.data;
};
