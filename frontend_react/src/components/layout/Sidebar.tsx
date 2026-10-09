import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Cpu,
  Eye,
  History,
  MessageSquare,
  Settings,
  ShieldCheck,
} from 'lucide-react';

interface SidebarProps {
  isOpen?: boolean;
  onClose?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ isOpen = false, onClose }) => {
  const anomalyNavItems = [
    { label: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
    { label: 'Models', path: '/models', icon: Cpu },
    { label: 'Inspect', path: '/inspect', icon: Eye },
    { label: 'History', path: '/history', icon: History },
    { label: 'Feedback', path: '/feedback', icon: MessageSquare },
    { label: 'Settings', path: '/settings', icon: Settings },
  ];

  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-industrial-950/70 backdrop-blur-xs z-40 lg:hidden transition-opacity"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`w-64 bg-industrial-900 border-r border-industrial-800 flex flex-col h-screen fixed left-0 top-0 z-50 select-none shadow-md transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Brand Header */}
        <div className="h-16 px-4 flex items-center justify-between border-b border-industrial-800/80 bg-industrial-950/40">
          <div className="flex items-center space-x-3 py-2">
            <div className="p-2 rounded-lg bg-brand-500 text-white shadow-sm flex items-center justify-center">
              <ShieldCheck className="w-5 h-5 stroke-[2.2]" />
            </div>
            <div>
              <span className="font-bold text-[14px] text-white tracking-tight block leading-none font-sans">
                Anomaly Detector
              </span>
              <span className="text-[10px] text-brand-300 font-mono tracking-wide mt-1 block">
                Visual Inspection Platform
              </span>
            </div>
          </div>
        </div>

        {/* Main Navigation List */}
        <nav className="flex-1 px-3 py-4 space-y-5 overflow-y-auto">
          {/* Navigation Section */}
          <div className="space-y-1">
            <div className="px-3 pb-1.5 text-[10px] font-mono uppercase tracking-wider text-industrial-400 font-bold">
              Navigation
            </div>
            {anomalyNavItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  onClick={onClose}
                  className={({ isActive }) =>
                    `flex items-center space-x-3 px-3 py-2 rounded-lg text-[13px] font-medium transition-all ${
                      isActive
                        ? 'bg-brand-500 text-white shadow-sm shadow-brand-500/20 font-semibold'
                        : 'text-industrial-300 hover:text-white hover:bg-industrial-800/70'
                    }`
                  }
                >
                  <Icon className="w-4 h-4 flex-shrink-0" />
                  <span>{item.label}</span>
                </NavLink>
              );
            })}
          </div>
        </nav>

        {/* Minimal Footer */}
        <div className="p-3 border-t border-industrial-800/80 bg-industrial-950/30 flex items-center justify-between text-[10px] text-industrial-400 font-mono font-medium">
          <span>Version 1.0.0</span>
          <span className="flex items-center space-x-1.5">
            <span className="w-2 h-2 rounded-full bg-pass-500 animate-pulse"></span>
            <span className="text-industrial-300">Ready</span>
          </span>
        </div>
      </aside>
    </>
  );
};
