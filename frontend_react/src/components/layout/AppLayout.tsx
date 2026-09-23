import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';

export const AppLayout: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);

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
