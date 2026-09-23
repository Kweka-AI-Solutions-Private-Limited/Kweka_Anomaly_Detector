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
  Leaf,
  Layers,
  ExternalLink,
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
        <div className="h-18 px-5 flex items-center justify-between border-b border-industrial-800/80 bg-industrial-950/40">
          <div className="flex items-center space-x-3.5 py-3">
            <div className="p-2.5 rounded-xl bg-brand-500 text-white shadow-sm flex items-center justify-center">
              <ShieldCheck className="w-6 h-6 stroke-[2.2]" />
            </div>
            <div>
              <span className="font-extrabold text-[17px] text-white tracking-tight block leading-none font-sans">
                Anomaly Detector
              </span>
              <span className="text-[11px] text-brand-300 font-mono tracking-wide mt-1 block">
                Visual Inspection Platform
              </span>
            </div>
          </div>
        </div>

        {/* Main Navigation List */}
        <nav className="flex-1 px-3.5 py-5 space-y-6 overflow-y-auto">
          {/* Applications Navigation Section */}
          <div className="space-y-1.5">
            <div className="px-3 pb-2 text-[11px] font-mono uppercase tracking-wider text-industrial-400 font-bold">
              Applications
            </div>
            
            {/* Anomaly Detection */}
            <NavLink
              to="/"
              end
              onClick={onClose}
              className={({ isActive }) =>
                `flex items-center space-x-3.5 px-4 py-3 rounded-xl text-[15px] font-semibold transition-all ${
                  isActive
                    ? 'bg-brand-500 text-white shadow-sm shadow-brand-500/20'
                    : 'text-industrial-300 hover:text-white hover:bg-industrial-800/70'
                }`
              }
            >
              <ShieldCheck className="w-5 h-5 flex-shrink-0" />
              <span>Anomaly Detection</span>
            </NavLink>

            {/* Leaf Disease Detection */}
            <NavLink
              to="/leaf-disease"
              onClick={onClose}
              className={({ isActive }) =>
                `flex items-center space-x-3.5 px-4 py-3 rounded-xl text-[15px] font-semibold transition-all ${
                  isActive
                    ? 'bg-brand-500 text-white shadow-sm shadow-brand-500/20'
                    : 'text-industrial-300 hover:text-white hover:bg-industrial-800/70'
                }`
              }
            >
              <Leaf className="w-5 h-5 flex-shrink-0" />
              <span>Leaf Disease Detection</span>
            </NavLink>

            {/* Pipe Counting (External Link) */}
            <a
              href="https://kweka-pipes-count.web.app/"
              target="_blank"
              rel="noopener noreferrer"
              onClick={onClose}
              className="flex items-center justify-between px-4 py-3 rounded-xl text-[15px] font-semibold text-industrial-300 hover:text-white hover:bg-industrial-800/70 transition-all group"
            >
              <div className="flex items-center space-x-3.5">
                <Layers className="w-5 h-5 flex-shrink-0" />
                <span>Pipe Counting</span>
              </div>
              <ExternalLink className="w-4 h-4 text-industrial-400 group-hover:text-white transition-colors" />
            </a>
          </div>

          {/* Anomaly Workspace Tools Section */}
          <div className="space-y-1.5 pt-4 border-t border-industrial-800/80">
            <div className="px-3 pb-2 text-[11px] font-mono uppercase tracking-wider text-industrial-400 font-bold">
              Anomaly Workspace
            </div>
            {anomalyNavItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  onClick={onClose}
                  className={({ isActive }) =>
                    `flex items-center space-x-3.5 px-4 py-3 rounded-xl text-[15px] font-semibold transition-all ${
                      isActive
                        ? 'bg-brand-500 text-white shadow-sm shadow-brand-500/20'
                        : 'text-industrial-300 hover:text-white hover:bg-industrial-800/70'
                    }`
                  }
                >
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  <span>{item.label}</span>
                </NavLink>
              );
            })}
          </div>
        </nav>

        {/* Minimal Footer */}
        <div className="p-4 border-t border-industrial-800/80 bg-industrial-950/30 flex items-center justify-between text-[11px] text-industrial-400 font-mono font-medium">
          <span>Version 1.0.0</span>
          <span className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-pass-500 animate-pulse"></span>
            <span className="text-industrial-300">Ready</span>
          </span>
        </div>
      </aside>
    </>
  );
};
