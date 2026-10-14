import React, { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { ShieldAlert, Key, ArrowRight, Lock, ExternalLink, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { exchangeAuthCode, getCurrentUser } from '../../api/auth';
import { Card } from '../common/Card';
import { Button } from '../common/Button';

const LOGIN_URL = 'https://agents.kweka.ai/login';

export const AppLayout: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);
  const [manualCodeInput, setManualCodeInput] = useState<string>('');
  const [isExchanging, setIsExchanging] = useState<boolean>(false);
  const [isCheckingAuth, setIsCheckingAuth] = useState<boolean>(true);
  const [authError, setAuthError] = useState<string | null>(null);
  const [redirectCountdown, setRedirectCountdown] = useState<number>(4);
  const [showManualForm, setShowManualForm] = useState<boolean>(false);

  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);

  // Initial authentication verification on component mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code =
      params.get('code') ||
      params.get('token') ||
      params.get('auth_code') ||
      params.get('exchange_code') ||
      params.get('jwt');
    const urlName =
      params.get('name') ||
      params.get('user_name') ||
      params.get('display_name') ||
      params.get('full_name');
    const urlEmail = params.get('email') || params.get('user_email') || params.get('mail');

    if (urlName && !/^[0-9a-fA-F]{16,}$/.test(urlName.trim())) {
      localStorage.setItem('user_name', urlName.trim());
    }
    if (urlEmail) {
      localStorage.setItem('user_email', urlEmail.trim());
    }

    if (code) {
      setIsCheckingAuth(true);
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
          params.delete('jwt');
          const newSearch = params.toString();
          const newUrl =
            window.location.pathname + (newSearch ? `?${newSearch}` : '') + window.location.hash;
          window.history.replaceState({}, '', newUrl);

          setIsAuthenticated(true);
          setAuthError(null);
          window.dispatchEvent(new Event('profile_updated'));
        })
        .catch((err) => {
          console.error('[Auth] Failed to verify exchange token:', err);
          setIsAuthenticated(false);
          setAuthError('You are not authorised to enter. Invalid or unrecognized authentication credentials.');
        })
        .finally(() => {
          setIsExchanging(false);
          setIsCheckingAuth(false);
        });
    } else {
      // Check stored token in localStorage
      const storedToken = localStorage.getItem('auth_token') || localStorage.getItem('token');
      if (storedToken) {
        setIsCheckingAuth(true);
        getCurrentUser()
          .then((res) => {
            if (res && res.authenticated) {
              if (res.name && !/^[0-9a-fA-F]{16,}$/.test(res.name.trim()) && !res.name.startsWith('Operator (')) {
                localStorage.setItem('user_name', res.name.trim());
              }
              if (res.email) {
                localStorage.setItem('user_email', res.email.trim());
              }
              window.dispatchEvent(new Event('profile_updated'));
              setIsAuthenticated(true);
              setAuthError(null);
            } else {
              throw new Error('User not authenticated');
            }
          })
          .catch((err) => {
            console.warn('[Auth] Stored session validation failed:', err);
            localStorage.removeItem('auth_token');
            localStorage.removeItem('token');
            localStorage.removeItem('user_id');
            localStorage.removeItem('user_name');
            localStorage.removeItem('user_email');
            localStorage.removeItem('user_profile');
            setIsAuthenticated(false);
            setAuthError('You are not authorised to enter. Session has expired.');
          })
          .finally(() => {
            setIsCheckingAuth(false);
          });
      } else {
        setIsAuthenticated(false);
        setAuthError('You are not authorised to enter.');
        setIsCheckingAuth(false);
      }
    }
  }, []);

  // Listen for 401 unauthorized events dispatched by axios interceptor
  useEffect(() => {
    const handleUnauthorized = (e: Event) => {
      const customEvent = e as CustomEvent<{ message?: string }>;
      const msg = customEvent.detail?.message || 'You are not authorised to enter';
      setIsAuthenticated(false);
      setAuthError(msg);
    };

    window.addEventListener('auth_unauthorized', handleUnauthorized);
    return () => window.removeEventListener('auth_unauthorized', handleUnauthorized);
  }, []);

  // Automatic redirect timer when restricted
  useEffect(() => {
    if (isCheckingAuth || isAuthenticated) return;

    setRedirectCountdown(4);
    const timer = setInterval(() => {
      setRedirectCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          window.location.href = LOGIN_URL;
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [isCheckingAuth, isAuthenticated]);

  const handleManualCodeSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!manualCodeInput.trim()) return;
    setIsExchanging(true);
    setAuthError(null);

    exchangeAuthCode(manualCodeInput.trim())
      .then((res) => {
        console.log('[Auth] Manual code exchange successful:', res.user_id);
        setIsAuthenticated(true);
        setAuthError(null);
        window.dispatchEvent(new Event('profile_updated'));
      })
      .catch(() => {
        setAuthError('You are not authorised to enter: Invalid authorization code or token.');
      })
      .finally(() => {
        setIsExchanging(false);
      });
  };

  // 1. Initial Authentication Verification Loading State
  if (isCheckingAuth) {
    return (
      <div className="min-h-screen w-screen bg-industrial-950 text-white flex items-center justify-center p-4">
        <div className="w-full max-w-sm text-center space-y-4">
          <div className="inline-flex items-center justify-center p-4 bg-brand-500/15 text-brand-400 rounded-2xl border border-brand-500/30 shadow-lg animate-pulse">
            <Lock className="w-10 h-10" />
          </div>
          <div className="space-y-1.5">
            <h2 className="text-lg font-bold font-sans text-white">Verifying Authorization</h2>
            <p className="text-xs text-industrial-400 font-mono">
              Validating cryptographic security credentials with Kweka Gateway...
            </p>
          </div>
          <div className="flex items-center justify-center space-x-2 text-xs font-mono text-brand-400">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Checking authentication status...</span>
          </div>
        </div>
      </div>
    );
  }

  // 2. Restricted Screen: User is NOT Authorised
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen w-screen bg-industrial-950 text-white flex items-center justify-center p-4">
        <div className="w-full max-w-md space-y-6">
          {/* Platform Branding */}
          <div className="text-center space-y-2">
            <div className="inline-flex items-center justify-center p-3.5 bg-rose-500/20 text-rose-400 rounded-2xl border border-rose-500/30 mb-1 shadow-lg">
              <ShieldAlert className="w-8 h-8" />
            </div>
            <h1 className="text-2xl font-bold tracking-tight font-sans text-white">
              Anomaly Detector
            </h1>
            <p className="text-xs text-industrial-400 font-mono">
              Industrial Visual Inspection & Defect Analysis
            </p>
          </div>

          <Card className="p-6 bg-industrial-900/95 border-industrial-800 space-y-5 shadow-2xl backdrop-blur-md">
            {/* Prominent Restriction Notice */}
            <div className="flex items-center space-x-2.5 text-rose-400 text-xs font-mono font-bold uppercase tracking-wider bg-rose-500/15 p-3 rounded-lg border border-rose-500/30">
              <ShieldAlert className="w-5 h-5 shrink-0 text-rose-400" />
              <span>You are not authorised to enter</span>
            </div>

            <p className="text-xs text-industrial-300 leading-relaxed">
              Access to this isolated industrial workstation requires an authorized, active session
              from the Kweka Central Agent Portal.
            </p>

            {authError && (
              <div className="text-xs font-mono text-rose-300 bg-rose-950/50 p-2.5 rounded border border-rose-800/60">
                {authError}
              </div>
            )}

            {/* Countdown Banner */}
            <div className="p-3 bg-industrial-950/80 rounded-lg border border-industrial-800 flex items-center justify-between text-xs font-mono">
              <span className="text-industrial-400">Redirecting to login portal in:</span>
              <span className="px-2 py-0.5 rounded bg-brand-500/20 text-brand-400 font-bold border border-brand-500/30">
                {redirectCountdown}s
              </span>
            </div>

            {/* Direct Login Redirection CTA */}
            <Button
              type="button"
              variant="primary"
              size="lg"
              className="w-full justify-center bg-brand-600 hover:bg-brand-500 text-white font-bold text-sm shadow-md"
              onClick={() => {
                window.location.href = LOGIN_URL;
              }}
              icon={<ExternalLink className="w-4 h-4" />}
            >
              Login via Kweka Portal
            </Button>

            {/* Manual Token Verification Toggle (Optional for testing) */}
            <div className="pt-2 border-t border-industrial-800/80">
              <button
                type="button"
                onClick={() => setShowManualForm((prev) => !prev)}
                className="w-full flex items-center justify-between text-[11px] font-mono text-industrial-400 hover:text-industrial-200 transition-colors py-1"
              >
                <span className="flex items-center space-x-1.5">
                  <Key className="w-3.5 h-3.5 text-industrial-500" />
                  <span>Have an authorized token? Enter manually</span>
                </span>
                {showManualForm ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>

              {showManualForm && (
                <form onSubmit={handleManualCodeSubmit} className="mt-3 space-y-2.5 animate-in fade-in">
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Paste JWT token or exchange code..."
                      value={manualCodeInput}
                      onChange={(e) => setManualCodeInput(e.target.value)}
                      className="w-full pl-3 pr-3 py-2 text-xs font-mono bg-industrial-950 text-white border border-industrial-700 rounded-md focus:border-brand-500 outline-none placeholder:text-industrial-600"
                    />
                  </div>

                  <Button
                    type="submit"
                    variant="secondary"
                    size="sm"
                    className="w-full justify-center text-xs"
                    isLoading={isExchanging}
                    icon={<ArrowRight className="w-3.5 h-3.5" />}
                  >
                    Verify Token & Enter
                  </Button>
                </form>
              )}
            </div>
          </Card>

          <p className="text-[10px] text-center text-industrial-500 font-mono">
            Kweka AI Solutions • Industrial Anomaly Detection Engine • User Isolated
          </p>
        </div>
      </div>
    );
  }

  // 3. Authenticated: Render Main Application Layout with Full User Isolation
  return (
    <div className="min-h-screen w-full min-w-0 max-w-full bg-industrial-100 flex overflow-x-hidden">
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

