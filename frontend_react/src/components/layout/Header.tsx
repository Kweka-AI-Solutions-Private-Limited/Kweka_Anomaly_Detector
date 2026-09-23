import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Bell, User, Settings, LogOut, ChevronDown, Menu } from 'lucide-react';
import {
  NotificationItem as NotificationModel,
  getNotifications,
  getUnreadNotificationCount,
  markNotificationRead,
  markAllNotificationsRead,
} from '../../api/notifications';
import { NotificationPanel } from '../notifications/NotificationPanel';

interface HeaderProps {
  onToggleMobileMenu?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onToggleMobileMenu }) => {
  const location = useLocation();
  const navigate = useNavigate();

  // Dropdown states
  const [isProfileOpen, setIsProfileOpen] = useState<boolean>(false);
  const [isNotificationOpen, setIsNotificationOpen] = useState<boolean>(false);

  // Notification data states
  const [unreadCount, setUnreadCount] = useState<number>(0);
  const [notifications, setNotifications] = useState<NotificationModel[]>([]);
  const [isLoadingNotifications, setIsLoadingNotifications] = useState<boolean>(false);
  const [notificationError, setNotificationError] = useState<string | null>(null);

  const profileRef = useRef<HTMLDivElement>(null);
  const notifContainerRef = useRef<HTMLDivElement>(null);

  // Fetch unread count from backend
  const fetchUnreadCount = useCallback(async () => {
    try {
      const count = await getUnreadNotificationCount();
      setUnreadCount(count);
    } catch (err) {
      console.warn('[WARN] Failed to fetch unread notification count:', err);
    }
  }, []);

  // Fetch full notification list from backend
  const fetchNotificationsList = useCallback(async () => {
    try {
      setNotificationError(null);
      const list = await getNotifications(20);
      setNotifications(list);
    } catch (err) {
      console.error('[ERROR] Failed to fetch notifications list:', err);
      setNotificationError('Failed to load notifications');
    }
  }, []);

  // Active-tab 12-second polling + visibilitychange handling
  useEffect(() => {
    fetchUnreadCount();

    const intervalId = setInterval(() => {
      if (document.visibilityState === 'visible') {
        fetchUnreadCount();
        if (isNotificationOpen) {
          fetchNotificationsList();
        }
      }
    }, 12000);

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        fetchUnreadCount();
        if (isNotificationOpen) {
          fetchNotificationsList();
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      clearInterval(intervalId);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [isNotificationOpen, fetchUnreadCount, fetchNotificationsList]);

  // Handle outside click for Profile dropdown
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (profileRef.current && !profileRef.current.contains(event.target as Node)) {
        setIsProfileOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Toggle notification panel
  const handleToggleNotifications = async () => {
    const nextState = !isNotificationOpen;
    setIsNotificationOpen(nextState);

    if (nextState) {
      setIsLoadingNotifications(true);
      await Promise.all([fetchNotificationsList(), fetchUnreadCount()]);
      setIsLoadingNotifications(false);
    }
  };

  // Mark single notification as read with optimistic update
  const handleMarkRead = async (id: string) => {
    const target = notifications.find((n) => n.id === id);
    if (!target || target.read) return;

    // Optimistic state update
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n))
    );
    setUnreadCount((prev) => Math.max(0, prev - 1));

    try {
      await markNotificationRead(id);
    } catch (err) {
      console.error(`[ERROR] Failed to mark notification ${id} as read:`, err);
      // Revert optimistic update on failure
      setNotifications((prev) =>
        prev.map((n) => (n.id === id ? { ...n, read: false } : n))
      );
      setUnreadCount((prev) => prev + 1);
      setNotificationError('Could not update notification state.');
    }
  };

  // Mark all notifications as read with optimistic update
  const handleMarkAllRead = async () => {
    if (unreadCount === 0) return;

    const previousNotifications = [...notifications];
    const previousUnreadCount = unreadCount;

    // Optimistic state update
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
    setUnreadCount(0);

    try {
      await markAllNotificationsRead();
    } catch (err) {
      console.error('[ERROR] Failed to mark all notifications as read:', err);
      // Revert on failure
      setNotifications(previousNotifications);
      setUnreadCount(previousUnreadCount);
      setNotificationError('Could not mark all notifications as read.');
    }
  };

  const getPageTitle = (path: string) => {
    switch (path) {
      case '/':
        return 'Product Overview';
      case '/dashboard':
        return 'Operational Dashboard';
      case '/models':
        return 'Inspection Models';
      case '/models/create':
        return 'Create Inspection Model';
      case '/inspect':
        return 'Inspection Workspace';
      case '/history':
        return 'Inspection History';
      case '/feedback':
        return 'Operator Feedback';
      case '/settings':
        return 'System Settings & Health';
      case '/leaf-disease':
        return 'Leaf Disease Detection';
      default:
        if (path.startsWith('/inspections/')) return 'Inspection Report Detail';
        return 'Product Overview';
    }
  };

  return (
    <header className="h-[70px] bg-white border-b border-industrial-200 px-4 sm:px-7 flex items-center justify-between flex-shrink-0 z-20 shadow-xs">
      {/* Left: Mobile Menu Button & Breadcrumb Title */}
      <div className="flex items-center space-x-3 text-xs font-mono">
        {onToggleMobileMenu && (
          <button
            type="button"
            onClick={onToggleMobileMenu}
            aria-label="Toggle Navigation Menu"
            className="lg:hidden p-2 text-industrial-600 hover:text-industrial-900 hover:bg-industrial-100 rounded-lg transition-colors"
          >
            <Menu className="w-5 h-5" />
          </button>
        )}
        <span className="text-industrial-400 uppercase tracking-wider font-bold text-xs hidden sm:inline">
          ANOMALY DETECTOR
        </span>
        <span className="text-industrial-300 hidden sm:inline">/</span>
        <span className="text-industrial-900 font-extrabold text-base sm:text-lg tracking-tight font-sans truncate">
          {getPageTitle(location.pathname)}
        </span>
      </div>

      {/* Right Controls */}
      <div className="flex items-center space-x-5">
        {/* Notifications Dropdown Container */}
        <div className="relative" ref={notifContainerRef}>
          <button
            type="button"
            onClick={handleToggleNotifications}
            aria-label={`Notifications, ${unreadCount} unread`}
            title={unreadCount > 0 ? `${unreadCount} unread notification(s)` : 'Notifications'}
            className="p-2.5 text-industrial-500 hover:text-industrial-900 hover:bg-industrial-100 rounded-full transition-colors relative focus:outline-none focus:ring-2 focus:ring-brand-500"
          >
            <Bell className="w-5 h-5" />
            {unreadCount > 0 && (
              <span className="absolute -top-0.5 -right-0.5 px-1.5 py-0.5 min-w-[18px] h-[18px] rounded-full bg-brand-500 text-white font-mono font-bold text-[10px] flex items-center justify-center ring-2 ring-white shadow-xs">
                {unreadCount > 99 ? '99+' : unreadCount}
              </span>
            )}
          </button>

          {/* Notification Dropdown Panel */}
          <NotificationPanel
            isOpen={isNotificationOpen}
            onClose={() => setIsNotificationOpen(false)}
            notifications={notifications}
            unreadCount={unreadCount}
            isLoading={isLoadingNotifications}
            error={notificationError}
            onRefresh={async () => {
              setIsLoadingNotifications(true);
              await Promise.all([fetchNotificationsList(), fetchUnreadCount()]);
              setIsLoadingNotifications(false);
            }}
            onMarkRead={handleMarkRead}
            onMarkAllRead={handleMarkAllRead}
          />
        </div>

        {/* Profile Avatar & Interactive Dropdown */}
        <div className="relative" ref={profileRef}>
          <button
            type="button"
            onClick={() => setIsProfileOpen((prev) => !prev)}
            className="flex items-center space-x-3 text-sm font-medium text-industrial-800 hover:bg-industrial-50 p-2 px-3 rounded-xl border border-industrial-200 transition-colors"
          >
            <div className="w-9 h-9 rounded-full bg-brand-500 text-white flex items-center justify-center font-mono font-bold text-sm shadow-sm">
              QA
            </div>
            <div className="text-left hidden sm:block">
              <p className="leading-tight font-extrabold text-base text-industrial-900">QA Operator</p>
              <p className="text-xs text-industrial-500 font-mono">Anomaly Detector</p>
            </div>
            <ChevronDown className="w-4 h-4 text-industrial-400" />
          </button>

          {/* Profile Dropdown Menu */}
          {isProfileOpen && (
            <div className="absolute right-0 mt-2 w-64 bg-white rounded-xl border border-industrial-200 shadow-xl py-2 z-50 text-base animate-in fade-in slide-in-from-top-2">
              <div className="px-5 py-3 border-b border-industrial-100">
                <p className="font-bold text-industrial-900 text-base">QA Operator</p>
                <p className="text-xs text-industrial-500 font-mono mt-0.5">
                  operator@anomalydetector.ai
                </p>
              </div>

              <div className="py-1.5">
                <button
                  type="button"
                  onClick={() => {
                    setIsProfileOpen(false);
                    navigate('/settings');
                  }}
                  className="w-full text-left px-5 py-2.5 hover:bg-industrial-50 text-industrial-800 flex items-center space-x-3 font-semibold text-sm"
                >
                  <User className="w-4.5 h-4.5 text-industrial-400" />
                  <span>Profile Overview</span>
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setIsProfileOpen(false);
                    navigate('/settings');
                  }}
                  className="w-full text-left px-5 py-2.5 hover:bg-industrial-50 text-industrial-800 flex items-center space-x-3 font-semibold text-sm"
                >
                  <Settings className="w-4.5 h-4.5 text-industrial-400" />
                  <span>System Settings</span>
                </button>
              </div>

              <div className="pt-1.5 border-t border-industrial-100">
                <button
                  type="button"
                  onClick={() => {
                    setIsProfileOpen(false);
                    alert('Logout action triggered.');
                  }}
                  className="w-full text-left px-5 py-2.5 hover:bg-reject-50 text-reject-600 font-semibold text-sm flex items-center space-x-3"
                >
                  <LogOut className="w-4.5 h-4.5" />
                  <span>Logout</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
