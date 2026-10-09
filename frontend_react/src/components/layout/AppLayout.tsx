import React, { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { ShieldAlert, Key, ArrowRight, Lock, Sparkles } from 'lucide-react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { exchangeAuthCode } from '../../api/auth';
import { Card } from '../common/Card';
import { Button } from '../common/Button';

export const AppLayout: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);
  const [manualCodeInput, setManualCodeInput] = useState<string>('');
  const [isExchanging, setIsExchanging] = useState<boolean>(false);
  const [authError, setAuthError] = useState<string | null>(null);

  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => {
    const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code') || params.get('token') || params.get('auth_code') || params.get('exchange_code');
    return !!(token || code);
  });

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code') || params.get('token') || params.get('auth_code') || params.get('exchange_code');
    const urlName = params.get('name') || params.get('user_name') || params.get('display_name') || params.get('full_name');
    const urlEmail = params.get('email') || params.get('user_email') || params.get('mail');

    if (urlName && !/^[0-9a-fA-F]{16,}$/.test(urlName.trim())) {
      localStorage.setItem('user_name', urlName.trim());
    }
    if (urlEmail) {
      localStorage.setItem('user_email', urlEmail.trim());
    }

    if (code) {
      setIsExchanging(true);
      exchangeAuthCode(code)
        .then((res) => {
          console.log('[Auth] Code exchange successful for user:', res.user_id);
          if (res.user?.name && !/^[0-9a-fA-F]{16,}$/.test(res.user.name.trim())) {
            localStorage.setItem('user_name', res.user.name.trim());
          }
          if (res.user?.email) {
            localStorage.setItem('user_email', res.user.email.trim());
          }
          params.delete('code');
          params.delete('token');
          params.delete('auth_code');
          params.delete('exchange_code');
          const newSearch = params.toString();
          const newUrl = window.location.pathname + (newSearch ? `?${newSearch}` : '') + window.location.hash;
          window.history.replaceState({}, '', newUrl);
          setIsAuthenticated(true);
          window.dispatchEvent(new Event('profile_updated'));
        })
        .catch((err) => {
          console.error('[Auth] Failed to exchange code:', err);
          setAuthError('Failed to verify exchange code. Please try again.');
        })
        .finally(() => {
          setIsExchanging(false);
        });
    } else if (urlName || urlEmail) {
      window.dispatchEvent(new Event('profile_updated'));
    }
  }, []);

  const handleManualCodeSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!manualCodeInput.trim()) return;
    setIsExchanging(true);
    setAuthError(null);

    exchangeAuthCode(manualCodeInput.trim())
      .then((res) => {
        console.log('[Auth] Manual code exchange successful:', res.user_id);
        setIsAuthenticated(true);
        window.dispatchEvent(new Event('profile_updated'));
      })
      .catch(() => {
        setAuthError('Invalid authorization code or token.');
      })
      .finally(() => {
        setIsExchanging(false);
      });
  };

  const handleDemoAccess = () => {
    localStorage.setItem('user_id', 'usr_default');
    localStorage.setItem('user_name', 'QA Operator');
    localStorage.setItem('user_email', 'operator@anomalydetector.ai');
    setIsAuthenticated(true);
    window.dispatchEvent(new Event('profile_updated'));
  };

  // Render Authentication Lock Screen for Incognito or Unauthenticated Sessions
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen w-screen bg-industrial-950 text-white flex items-center justify-center p-4">
        <div className="w-full max-w-md space-y-6">
          {/* Platform Branding */}
          <div className="text-center space-y-2">
            <div className="inline-flex items-center justify-center p-3 bg-brand-600/20 text-brand-500 rounded-xl border border-brand-500/30 mb-2">
              <Lock className="w-8 h-8" />
            </div>
            <h1 className="text-2xl font-bold tracking-tight font-sans text-white">
              Anomaly Detector
            </h1>
            <p className="text-xs text-industrial-400 font-mono">
              Industrial Visual Inspection & Defect Analysis
            </p>
          </div>

          <Card className="p-6 bg-industrial-900/90 border-industrial-800 space-y-5 shadow-2xl backdrop-blur">
            <div className="flex items-center space-x-2 text-amber-400 text-xs font-mono font-bold uppercase tracking-wider bg-amber-500/10 p-2.5 rounded border border-amber-500/20">
              <ShieldAlert className="w-4 h-4 shrink-0 text-amber-400" />
              <span>Authentication Required</span>
            </div>

            <p className="text-xs text-industrial-300 leading-relaxed">
              You are accessing an isolated industrial workstation in Incognito / Unauthenticated mode. An exchange token or active portal session is required.
            </p>

            {authError && (
              <div className="text-xs font-mono text-rose-400 bg-rose-500/10 p-2.5 rounded border border-rose-500/20">
                {authError}
              </div>
            )}

            {/* Manual Code Input Form */}
            <form onSubmit={handleManualCodeSubmit} className="space-y-3">
              <div>
                <label className="block text-[10px] font-mono font-bold text-industrial-400 uppercase mb-1">
                  Authorization / Exchange Code
                </label>
                <div className="relative">
                  <Key className="w-4 h-4 absolute left-3 top-2.5 text-industrial-500" />
                  <input
                    type="text"
                    placeholder="Paste exchange code or token..."
                    value={manualCodeInput}
                    onChange={(e) => setManualCodeInput(e.target.value)}
                    className="w-full pl-9 pr-3 py-2 text-xs font-mono bg-industrial-950 text-white border border-industrial-700 rounded focus:border-brand-500 outline-none"
                  />
                </div>
              </div>

              <Button
                type="submit"
                variant="primary"
                size="md"
                className="w-full justify-center"
                isLoading={isExchanging}
                icon={<ArrowRight className="w-4 h-4" />}
              >
                Verify & Authenticate
              </Button>
            </form>

            <div className="relative flex items-center justify-center my-4">
              <div className="border-t border-industrial-800 w-full" />
              <span className="bg-industrial-900 px-3 text-[10px] font-mono text-industrial-500 uppercase absolute">
                OR
              </span>
            </div>

            {/* Central Portal Redirect & Demo Access */}
            <div className="space-y-2">
              <a
                href="https://kw-prototypes-service-238644809220.asia-south1.run.app"
                target="_blank"
                rel="noreferrer"
                className="w-full inline-flex items-center justify-center px-4 py-2 text-xs font-bold font-mono text-industrial-200 bg-industrial-800 hover:bg-industrial-700 rounded border border-industrial-700 transition-colors"
              >
                Login via Kweka Portal
              </a>

              <button
                type="button"
                onClick={handleDemoAccess}
                className="w-full inline-flex items-center justify-center space-x-1.5 px-4 py-2 text-xs font-mono text-industrial-400 hover:text-white transition-colors"
              >
                <Sparkles className="w-3.5 h-3.5 text-brand-400" />
                <span>Continue as Demo Operator (QA Testing)</span>
              </button>
            </div>
          </Card>

          <p className="text-[10px] text-center text-industrial-500 font-mono">
            Kweka AI Solutions • Industrial Anomaly Detection Engine v1.0
          </p>
        </div>
      </div>
    );
  }

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
