import { apiClient } from './client';

export type NotificationSeverity = 'success' | 'warning' | 'error' | 'info';

export type NotificationType =
  | 'MODEL_BUILD_COMPLETED'
  | 'MODEL_BUILD_FAILED'
  | 'INSPECTION_RUN_COMPLETED'
  | 'INSPECTION_RUN_PARTIAL'
  | 'INSPECTION_RUN_FAILED';

export interface NotificationItem {
  id: string;
  type: NotificationType | string;
  title: string;
  message: string;
  severity: NotificationSeverity;
  related_model_id?: string | null;
  related_version_id?: string | null;
  related_run_id?: string | null;
  target_route: string;
  read: boolean;
  idempotency_key: string;
  created_at: string;
}

export interface UnreadCountResponse {
  unread_count: number;
}

export interface MarkAllReadResponse {
  message: string;
  modified_count: number;
}

export async function getNotifications(limit = 20, unreadOnly = false): Promise<NotificationItem[]> {
  const response = await apiClient.get<NotificationItem[]>('/api/notifications', {
    params: { limit, unread_only: unreadOnly },
  });
  return response.data;
}

export async function getUnreadNotificationCount(): Promise<number> {
  const response = await apiClient.get<UnreadCountResponse>('/api/notifications/unread-count');
  return response.data.unread_count;
}

export async function markNotificationRead(notificationId: string): Promise<NotificationItem> {
  const response = await apiClient.patch<NotificationItem>(`/api/notifications/${notificationId}/read`);
  return response.data;
}

export async function markAllNotificationsRead(): Promise<MarkAllReadResponse> {
  const response = await apiClient.patch<MarkAllReadResponse>('/api/notifications/read-all');
  return response.data;
}
