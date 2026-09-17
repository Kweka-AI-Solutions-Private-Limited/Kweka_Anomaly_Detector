import React from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';

export const AppLayout: React.FC = () => {
  return (
    <div className="min-h-screen w-screen bg-industrial-100 flex overflow-x-hidden">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Layout Area */}
      <div className="flex-1 ml-64 flex flex-col min-h-screen min-w-0">
        <Header />
        
        {/* Page Content Container */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto bg-industrial-100/70">
          <div className="w-full max-w-[1920px] mx-auto space-y-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};
