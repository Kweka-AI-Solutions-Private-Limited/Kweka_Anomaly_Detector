import React, { useEffect, useRef } from 'react';
import { CheckCheck, RefreshCw, BellOff, AlertCircle } from 'lucide-react';
import { NotificationItem as NotificationModel } from '../../api/notifications';
import { NotificationItem } from './NotificationItem';

interface NotificationPanelProps {
  isOpen: boolean;
  onClose: () => void;
  notifications: NotificationModel[];
  unreadCount: number;
  isLoading: boolean;
  error: string | null;
  onRefresh: () => void;
  onMarkRead: (id: string) => void;
  onMarkAllRead: () => void;
}

export const NotificationPanel: React.FC<NotificationPanelProps> = ({
  isOpen,
  onClose,
  notifications,
  unreadCount,
  isLoading,
  error,
  onRefresh,
  onMarkRead,
  onMarkAllRead,
}) => {
  const panelRef = useRef<HTMLDivElement>(null);

  // Close panel on outside click or Escape key press
  useEffect(() => {
    if (!isOpen) return;

    function handleClickOutside(event: MouseEvent) {
      if (panelRef.current && !panelRef.current.contains(event.target as Node)) {
        onClose();
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        onClose();
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      ref={panelRef}
      role="dialog"
      aria-label="Notifications Panel"
      className="absolute right-0 mt-2 w-[360px] sm:w-[400px] max-w-[calc(100vw-32px)] bg-white rounded-xl border border-industrial-200 shadow-2xl z-50 overflow-hidden animate-in fade-in slide-in-from-top-2 text-industrial-900"
    >
      {/* Panel Header */}
      <div className="p-3.5 px-4 bg-industrial-50 border-b border-industrial-200 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <h3 className="font-mono font-black text-xs uppercase tracking-wider text-industrial-900">
            Notifications
          </h3>
          {unreadCount > 0 && (
            <span className="px-2 py-0.5 rounded-full bg-brand-500 text-white font-mono font-bold text-[10px]">
              {unreadCount > 99 ? '99+' : unreadCount} new
            </span>
          )}
        </div>

        {unreadCount > 0 && (
          <button
            type="button"
            onClick={onMarkAllRead}
            className="text-xs font-mono font-bold text-brand-600 hover:text-brand-800 flex items-center space-x-1.5 transition-colors focus:outline-none focus:ring-1 focus:ring-brand-500 rounded px-1.5 py-0.5"
            title="Mark all notifications as read"
          >
            <CheckCheck className="w-3.5 h-3.5" />
            <span>Mark all read</span>
          </button>
        )}
      </div>

      {/* Panel Content Body */}
      <div className="max-h-[380px] overflow-y-auto divide-y divide-industrial-100 font-sans">
        {isLoading && notifications.length === 0 ? (
          /* Loading Skeleton */
          <div className="p-4 space-y-3">
            {[1, 2, 3].map((n) => (
              <div key={n} className="animate-pulse flex items-start space-x-3">
                <div className="w-4.5 h-4.5 rounded-full bg-industrial-200 mt-1 flex-shrink-0" />
                <div className="flex-1 space-y-2">
                  <div className="h-3.5 bg-industrial-200 rounded w-3/4" />
                  <div className="h-3 bg-industrial-100 rounded w-5/6" />
                </div>
              </div>
            ))}
          </div>
        ) : error ? (
          /* Error State */
          <div className="p-6 text-center space-y-2 font-mono text-xs">
            <AlertCircle className="w-6 h-6 text-rose-500 mx-auto" />
            <p className="font-bold text-industrial-800">Couldn't load notifications.</p>
            <p className="text-[11px] text-industrial-500 font-sans">
              Check your network connection and try again.
            </p>
            <button
              type="button"
              onClick={onRefresh}
              className="mt-2 inline-flex items-center space-x-1.5 px-3 py-1.5 bg-industrial-100 hover:bg-industrial-200 text-industrial-800 font-bold rounded-lg transition-colors text-xs"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Retry</span>
            </button>
          </div>
        ) : notifications.length === 0 ? (
          /* Empty State */
          <div className="p-8 text-center space-y-2">
            <BellOff className="w-8 h-8 text-industrial-300 mx-auto" />
            <p className="font-mono font-bold text-xs text-industrial-800 uppercase tracking-wider">
              You're all caught up
            </p>
            <p className="text-xs text-industrial-500 font-sans">
              No new completion notifications at this time.
            </p>
          </div>
        ) : (
          /* Notifications List */
          notifications.map((item) => (
            <NotificationItem
              key={item.id}
              notification={item}
              onMarkRead={onMarkRead}
              onClosePanel={onClose}
            />
          ))
        )}
      </div>

      {/* Footer Status Bar */}
      <div className="p-2.5 px-4 bg-industrial-50/50 border-t border-industrial-100 flex items-center justify-between text-[10px] font-mono text-industrial-400">
        <span>Auto-sync active</span>
        {notifications.length > 0 && (
          <span>Showing latest {notifications.length}</span>
        )}
      </div>
    </div>
  );
};
