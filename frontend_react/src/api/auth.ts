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
    if (response.data.user?.name && response.data.user.name !== response.data.user_id) {
      localStorage.setItem('user_name', response.data.user.name);
    }
    if (response.data.user?.email) {
      localStorage.setItem('user_email', response.data.user.email);
    }
  }
  return response.data;
}

export async function getCurrentUser(): Promise<{ user_id: string; name?: string; email?: string; authenticated: boolean }> {
  const response = await apiClient.get('/api/auth/me');
  return response.data;
}
