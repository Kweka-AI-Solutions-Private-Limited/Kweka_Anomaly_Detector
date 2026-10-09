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
    const token = response.data.access_token;
    const userId = response.data.user_id;
    const userObj = response.data.user || {};

    localStorage.setItem('auth_token', token);
    localStorage.setItem('token', token);
    localStorage.setItem('user_id', userId);

    const firstName = userObj.first_name || '';
    const lastName = userObj.last_name || '';
    const combinedName = `${firstName} ${lastName}`.trim();
    const name = combinedName || userObj.name || userObj.full_name || userObj.display_name;

    if (name && name !== userId && !/^[0-9a-fA-F]{16,}$/.test(name)) {
      localStorage.setItem('user_name', name);
    }
    if (userObj.email) {
      localStorage.setItem('user_email', userObj.email);
    }

    localStorage.setItem('user_profile', JSON.stringify({
      id: userId,
      name: name || undefined,
      email: userObj.email || undefined,
      first_name: firstName || undefined,
      last_name: lastName || undefined,
      role: userObj.role || 'user'
    }));
  }
  return response.data;
}

export async function getCurrentUser(): Promise<{ user_id: string; name?: string; email?: string; authenticated: boolean }> {
  const response = await apiClient.get('/api/auth/me');
  return response.data;
}
