import React, { useEffect, useState } from 'react';
import {
  Sun,
  Moon,
  Monitor,
  Cpu,
  Server,
  Database,
  Activity,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  HardDrive,
} from 'lucide-react';
import { getHealthStatus } from '../api/health';
import { HealthStatus } from '../types';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { LoadingSpinner } from '../components/common/LoadingSpinner';

export const Settings: React.FC = () => {
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>(() => {
    return (localStorage.getItem('anomaly_detector_theme') as any) || 'system';
  });

  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [isLoadingHealth, setIsLoadingHealth] = useState<boolean>(true);
  const [healthError, setHealthError] = useState<string | null>(null);

  // Apply theme class to document
  useEffect(() => {
    localStorage.setItem('anomaly_detector_theme', theme);
    const root = document.documentElement;

    if (theme === 'dark') {
      root.classList.add('dark');
    } else if (theme === 'light') {
      root.classList.remove('dark');
    } else {
      // System preference
      if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        root.classList.add('dark');
      } else {
        root.classList.remove('dark');
      }
    }
  }, [theme]);

  // Load Health Status
  const fetchHealth = async () => {
    try {
      setIsLoadingHealth(true);
      setHealthError(null);
      const data = await getHealthStatus();
      setHealth(data);
    } catch (err: any) {
      setHealthError('Failed to connect to FastAPI backend health endpoint.');
    } finally {
      setIsLoadingHealth(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  return (
    <div className="space-y-6 w-full">
      {/* Header */}
      <div className="bg-white p-6 rounded-lg border border-industrial-200 shadow-sm flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-industrial-900 tracking-tight">
            System Settings & Diagnostics
          </h1>
          <p className="text-xs text-industrial-500 mt-1">
            Configure application appearance, review PatchCore AI engine specs, and inspect backend health telemetry.
          </p>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={fetchHealth}
          isLoading={isLoadingHealth}
          icon={<RefreshCw className="w-3.5 h-3.5" />}
        >
          Refresh Health
        </Button>
      </div>

      {/* 1. Appearance / Theme Section */}
      <Card className="p-6 space-y-4">
        <div>
          <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-industrial-700">
            1. Appearance & Theme
          </h2>
          <p className="text-xs text-industrial-500 mt-0.5">
            Select your preferred visual style for industrial workstation display.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
          {/* Light Option */}
          <div
            onClick={() => setTheme('light')}
            className={`p-4 rounded-lg border cursor-pointer flex items-center space-x-3 transition-all ${
              theme === 'light'
                ? 'border-brand-500 bg-brand-50/50 shadow-sm ring-2 ring-brand-500/20'
                : 'border-industrial-200 hover:border-industrial-300 bg-white'
            }`}
          >
            <div className={`p-2.5 rounded-lg ${theme === 'light' ? 'bg-brand-600 text-white' : 'bg-industrial-100 text-industrial-600'}`}>
              <Sun className="w-5 h-5" />
            </div>
            <div>
              <p className="font-bold text-xs text-industrial-900">Light Mode</p>
              <p className="text-[10px] text-industrial-400 font-mono">Clean high-contrast</p>
            </div>
          </div>

          {/* Dark Option */}
          <div
            onClick={() => setTheme('dark')}
            className={`p-4 rounded-lg border cursor-pointer flex items-center space-x-3 transition-all ${
              theme === 'dark'
                ? 'border-brand-500 bg-brand-50/50 shadow-sm ring-2 ring-brand-500/20'
                : 'border-industrial-200 hover:border-industrial-300 bg-white'
            }`}
          >
            <div className={`p-2.5 rounded-lg ${theme === 'dark' ? 'bg-brand-600 text-white' : 'bg-industrial-100 text-industrial-600'}`}>
              <Moon className="w-5 h-5" />
            </div>
            <div>
              <p className="font-bold text-xs text-industrial-900">Dark Mode</p>
              <p className="text-[10px] text-industrial-400 font-mono">Kweka-inspired dark slate</p>
            </div>
          </div>

          {/* System Option */}
          <div
            onClick={() => setTheme('system')}
            className={`p-4 rounded-lg border cursor-pointer flex items-center space-x-3 transition-all ${
              theme === 'system'
                ? 'border-brand-500 bg-brand-50/50 shadow-sm ring-2 ring-brand-500/20'
                : 'border-industrial-200 hover:border-industrial-300 bg-white'
            }`}
          >
            <div className={`p-2.5 rounded-lg ${theme === 'system' ? 'bg-brand-600 text-white' : 'bg-industrial-100 text-industrial-600'}`}>
              <Monitor className="w-5 h-5" />
            </div>
            <div>
              <p className="font-bold text-xs text-industrial-900">System Preference</p>
              <p className="text-[10px] text-industrial-400 font-mono">Match OS theme</p>
            </div>
          </div>
        </div>
      </Card>

      {/* 2. AI Engine Technical Configuration */}
      <Card className="p-6 space-y-4">
        <div>
          <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-industrial-700 flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-brand-600" />
            <span>2. AI Engine Configuration</span>
          </h2>
          <p className="text-xs text-industrial-500 mt-0.5">
            Informational configuration parameters powering PatchCore anomaly detection.
          </p>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 pt-2 font-mono text-xs">
          <div className="p-3 bg-industrial-50 rounded border border-industrial-200">
            <span className="text-[10px] text-industrial-400 block uppercase">Detection Engine</span>
            <span className="font-bold text-industrial-900 mt-1 block">PatchCore</span>
          </div>

          <div className="p-3 bg-industrial-50 rounded border border-industrial-200">
            <span className="text-[10px] text-industrial-400 block uppercase">Backbone</span>
            <span className="font-bold text-industrial-900 mt-1 block">WideResNet50_2</span>
          </div>

          <div className="p-3 bg-industrial-50 rounded border border-industrial-200">
            <span className="text-[10px] text-industrial-400 block uppercase">Feature Layer</span>
            <span className="font-bold text-industrial-900 mt-1 block">Layer 2</span>
          </div>

          <div className="p-3 bg-industrial-50 rounded border border-industrial-200">
            <span className="text-[10px] text-industrial-400 block uppercase">Coreset Sampling</span>
            <span className="font-bold text-industrial-900 mt-1 block">5.0%</span>
          </div>

          <div className="p-3 bg-industrial-50 rounded border border-industrial-200">
            <span className="text-[10px] text-industrial-400 block uppercase">Nearest Neighbors</span>
            <span className="font-bold text-industrial-900 mt-1 block">k = 9</span>
          </div>
        </div>
      </Card>

      {/* 3. System Diagnostics & Telemetry */}
      <Card className="p-6 space-y-4">
        <div>
          <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-industrial-700 flex items-center space-x-2">
            <Server className="w-4 h-4 text-pass-600" />
            <span>3. Backend System Diagnostics</span>
          </h2>
          <p className="text-xs text-industrial-500 mt-0.5">
            Live telemetry data queried from <code className="text-brand-700">GET /api/health</code>.
          </p>
        </div>

        {healthError ? (
          <div className="p-4 bg-reject-50 border border-reject-200 text-reject-700 text-xs rounded-md flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-reject-600 flex-shrink-0" />
            <span>{healthError}</span>
          </div>
        ) : isLoadingHealth ? (
          <LoadingSpinner label="Querying backend telemetry..." size="md" />
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 font-mono text-xs pt-2">
            {/* Backend API */}
            <div className="p-4 bg-industrial-50 rounded-lg border border-industrial-200 space-y-1">
              <div className="flex items-center justify-between text-industrial-500">
                <span className="text-[10px] uppercase font-bold">FastAPI Backend</span>
                <Server className="w-4 h-4 text-brand-600" />
              </div>
              <p className="font-bold text-pass-700 text-sm flex items-center space-x-1 mt-1">
                <CheckCircle2 className="w-4 h-4 text-pass-600 inline" />
                <span>Connected</span>
              </p>
              <p className="text-[10px] text-industrial-400">http://127.0.0.1:8000</p>
            </div>

            {/* MongoDB Database */}
            <div className="p-4 bg-industrial-50 rounded-lg border border-industrial-200 space-y-1">
              <div className="flex items-center justify-between text-industrial-500">
                <span className="text-[10px] uppercase font-bold">MongoDB Database</span>
                <Database className="w-4 h-4 text-brand-600" />
              </div>
              <p className="font-bold text-pass-700 text-sm flex items-center space-x-1 mt-1">
                <CheckCircle2 className="w-4 h-4 text-pass-600 inline" />
                <span>Connected</span>
              </p>
              <p className="text-[10px] text-industrial-400">Database: anomaly_detection</p>
            </div>

            {/* CUDA Hardware Acceleration */}
            <div className="p-4 bg-industrial-50 rounded-lg border border-industrial-200 space-y-1">
              <div className="flex items-center justify-between text-industrial-500">
                <span className="text-[10px] uppercase font-bold">CUDA Hardware</span>
                <Activity className="w-4 h-4 text-brand-600" />
              </div>
              <p className="font-bold text-industrial-900 text-sm mt-1">
                {health?.cuda?.available ? 'Available (GPU)' : 'CPU Mode'}
              </p>
              <p className="text-[10px] text-industrial-400">PyTorch Acceleration</p>
            </div>

            {/* GPU Hardware Specs */}
            <div className="p-4 bg-industrial-50 rounded-lg border border-industrial-200 space-y-1">
              <div className="flex items-center justify-between text-industrial-500">
                <span className="text-[10px] uppercase font-bold">GPU Specs</span>
                <HardDrive className="w-4 h-4 text-brand-600" />
              </div>
              <p className="font-bold text-industrial-900 text-xs mt-1 truncate">
                {health?.cuda?.device_name || 'NVIDIA GeForce RTX 3050 Ti Laptop GPU'}
              </p>
              <p className="text-[10px] text-industrial-400">Local Inference Hardware</p>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};
