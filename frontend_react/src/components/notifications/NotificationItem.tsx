import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Cpu,
  Play,
  Info
} from 'lucide-react';
import { NotificationItem as NotificationModel } from '../../api/notifications';

interface NotificationItemProps {
  notification: NotificationModel;
  onMarkRead: (id: string) => void;
  onClosePanel: () => void;
}

export function formatRelativeTime(dateString: string): string {
  if (!dateString) return 'Just now';
  const date = new Date(dateString);
  const now = new Date();
  const diffInMs = now.getTime() - date.getTime();
  const diffInSecs = Math.floor(diffInMs / 1000);
  const diffInMins = Math.floor(diffInSecs / 60);
  const diffInHours = Math.floor(diffInMins / 60);
  const diffInDays = Math.floor(diffInHours / 24);

  if (diffInSecs < 45) return 'Just now';
  if (diffInMins < 60) return `${diffInMins} min${diffInMins > 1 ? 's' : ''} ago`;
  if (diffInHours < 24) return `${diffInHours} hr${diffInHours > 1 ? 's' : ''} ago`;
  if (diffInDays === 1) return 'Yesterday';
  if (diffInDays < 7) return `${diffInDays} days ago`;

  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export const NotificationItem: React.FC<NotificationItemProps> = ({
  notification,
  onMarkRead,
  onClosePanel,
}) => {
  const navigate = useNavigate();

  const getIcon = () => {
    switch (notification.type) {
      case 'MODEL_BUILD_COMPLETED':
        return <CheckCircle2 className="w-4.5 h-4.5 text-emerald-600 flex-shrink-0" />;
      case 'MODEL_BUILD_FAILED':
        return <XCircle className="w-4.5 h-4.5 text-rose-600 flex-shrink-0" />;
      case 'INSPECTION_RUN_COMPLETED':
        return <Play className="w-4.5 h-4.5 text-brand-600 flex-shrink-0" />;
      case 'INSPECTION_RUN_PARTIAL':
        return <AlertTriangle className="w-4.5 h-4.5 text-amber-600 flex-shrink-0" />;
      case 'INSPECTION_RUN_FAILED':
        return <XCircle className="w-4.5 h-4.5 text-rose-600 flex-shrink-0" />;
      default:
        if (notification.severity === 'error') {
          return <XCircle className="w-4.5 h-4.5 text-rose-600 flex-shrink-0" />;
        }
        if (notification.severity === 'warning') {
          return <AlertTriangle className="w-4.5 h-4.5 text-amber-600 flex-shrink-0" />;
        }
        if (notification.severity === 'success') {
          return <CheckCircle2 className="w-4.5 h-4.5 text-emerald-600 flex-shrink-0" />;
        }
        return <Info className="w-4.5 h-4.5 text-industrial-500 flex-shrink-0" />;
    }
  };

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();

    // Mark as read if currently unread
    if (!notification.read) {
      onMarkRead(notification.id);
    }

    // Navigate if target route exists and is non-empty
    if (notification.target_route && notification.target_route.trim() !== '') {
      onClosePanel();
      navigate(notification.target_route);
    }
  };

  return (
    <div
      onClick={handleClick}
      className={`p-3.5 px-4 transition-colors cursor-pointer border-b border-industrial-100 last:border-b-0 hover:bg-industrial-50/80 flex items-start space-x-3 text-left ${
        !notification.read
          ? 'bg-brand-50/40 border-l-4 border-l-brand-500 font-medium'
          : 'bg-white opacity-85'
      }`}
    >
      <div className="mt-0.5">{getIcon()}</div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <p
            className={`text-xs font-sans truncate ${
              !notification.read
                ? 'font-bold text-industrial-900'
                : 'font-semibold text-industrial-700'
            }`}
          >
            {notification.title}
          </p>
          <span className="text-[10px] font-mono text-industrial-400 flex-shrink-0">
            {formatRelativeTime(notification.created_at)}
          </span>
        </div>

        <p className="text-xs text-industrial-600 font-sans mt-0.5 line-clamp-2 leading-relaxed">
          {notification.message}
        </p>

        {!notification.read && (
          <div className="mt-1.5 flex items-center space-x-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-500" />
            <span className="text-[10px] font-mono font-bold text-brand-600 uppercase tracking-wider">
              Unread
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
