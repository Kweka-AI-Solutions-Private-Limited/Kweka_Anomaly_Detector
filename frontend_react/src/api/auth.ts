import { apiClient } from './client';

export interface AuthTokenResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  user: Record<string, any>;
}

export async function exchangeAuthCode(code: string): Promise<AuthTokenResponse> {
  const response = await apiClient.post<AuthTokenResponse>('/api/auth/exchange', { code });
  if (response.data.access_token) {
    localStorage.setItem('auth_token', response.data.access_token);
    localStorage.setItem('user_id', response.data.user_id);
  }
  return response.data;
}

export async function getCurrentUser(): Promise<{ user_id: string; authenticated: boolean }> {
  const response = await apiClient.get('/api/auth/me');
  return response.data;
}
