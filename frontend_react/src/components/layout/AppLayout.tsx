import React, { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { exchangeAuthCode } from '../../api/auth';

export const AppLayout: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code') || params.get('token') || params.get('auth_code') || params.get('exchange_code');
    if (code) {
      exchangeAuthCode(code)
        .then((res) => {
          console.log('[Auth] Code exchange successful for user:', res.user_id);
          params.delete('code');
          params.delete('token');
          params.delete('auth_code');
          params.delete('exchange_code');
          const newSearch = params.toString();
          const newUrl = window.location.pathname + (newSearch ? `?${newSearch}` : '') + window.location.hash;
          window.history.replaceState({}, '', newUrl);
        })
        .catch((err) => {
          console.error('[Auth] Failed to exchange code:', err);
        });
    }
  }, []);

  return (
    <div className="min-h-screen w-screen bg-industrial-100 flex overflow-x-hidden">
      {/* Responsive Sidebar */}
      <Sidebar isOpen={isMobileMenuOpen} onClose={() => setIsMobileMenuOpen(false)} />

      {/* Main Layout Area */}
      <div className="flex-1 ml-0 lg:ml-64 flex flex-col min-h-screen min-w-0 transition-all duration-300">
        <Header onToggleMobileMenu={() => setIsMobileMenuOpen((prev) => !prev)} />
        
        {/* Page Content Container */}
        <main className="flex-1 p-3 sm:p-6 lg:p-8 overflow-y-auto bg-industrial-100/70">
          <div className="w-full max-w-[1920px] mx-auto space-y-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};
