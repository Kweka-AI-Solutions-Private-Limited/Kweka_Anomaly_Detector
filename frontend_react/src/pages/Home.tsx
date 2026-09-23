import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ShieldCheck,
  Cpu,
  Layers,
  Play,
  Plus,
  ArrowRight,
  CheckCircle2,
  XCircle,
  TrendingUp,
  Activity,
  Zap,
  Eye,
  Target,
  Sparkles,
  HelpCircle,
  FileCheck,
  AlertTriangle,
  Factory,
  CircuitBoard,
  Shirt,
  Package,
  Building2,
  ChevronRight,
  Database,
  Search,
  Sliders,
  Check,
  RefreshCw,
} from 'lucide-react';
import { getDashboardSummary, DashboardSummaryResponse } from '../api/dashboard';
import { getModels } from '../api/models';
import { Model } from '../types';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { Badge } from '../components/common/Badge';
import { LoadingSpinner } from '../components/common/LoadingSpinner';

// Animated Count-Up Hook respecting prefers-reduced-motion
function useCountUp(endValue: number, durationMs: number = 1000) {
  const [count, setCount] = useState<number>(0);

  useEffect(() => {
    // Respect reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion || durationMs <= 0 || endValue === 0) {
      setCount(endValue);
      return;
    }

    let startTime: number | null = null;
    let animationFrameId: number;

    const step = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / durationMs, 1);
      // Ease-out quad
      const easedProgress = 1 - (1 - progress) * (1 - progress);
      setCount(Math.round(easedProgress * endValue));

      if (progress < 1) {
        animationFrameId = requestAnimationFrame(step);
      }
    };

    animationFrameId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animationFrameId);
  }, [endValue, durationMs]);

  return count;
}

export const Home: React.FC = () => {
  const navigate = useNavigate();

  const [dashboardData, setDashboardData] = useState<DashboardSummaryResponse | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [activePipelineStep, setActivePipelineStep] = useState<number>(0);

  // Cycle pipeline step automatically if motion is enabled
  useEffect(() => {
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const interval = setInterval(() => {
      setActivePipelineStep((prev) => (prev + 1) % 6);
    }, 3500);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    let isMounted = true;

    async function loadData() {
      try {
        setIsLoading(true);
        const [dashRes, modelList] = await Promise.allSettled([
          getDashboardSummary(),
          getModels(),
        ]);

        if (!isMounted) return;

        if (dashRes.status === 'fulfilled') {
          setDashboardData(dashRes.value);
        }

        if (modelList.status === 'fulfilled') {
          // Sort models by created_at descending (recent-first)
          const sorted = [...modelList.value].sort((a, b) => {
            const timeA = new Date(a.created_at || 0).getTime();
            const timeB = new Date(b.created_at || 0).getTime();
            return timeB - timeA;
          });
          setModels(sorted);
        }
      } catch (err) {
        console.error('Error fetching homepage dynamic data:', err);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }

    loadData();

    return () => {
      isMounted = false;
    };
  }, []);

  const totalModelsCount = models.length;
  const totalInspectionsCount = dashboardData?.kpi.total_inspections ?? 0;
  const totalRunsCount = dashboardData?.kpi.total_runs ?? 0;
  const activeModelsCount = dashboardData?.kpi.active_models ?? models.filter((m) => m.status === 'active').length;
  const passRateVal = dashboardData?.kpi.pass_rate ?? 0;
  const rejectCountVal = dashboardData?.kpi.reject_count ?? 0;

  // Animated Count-up values
  const animatedModels = useCountUp(totalModelsCount, 1200);
  const animatedInspections = useCountUp(totalInspectionsCount, 1200);
  const animatedRuns = useCountUp(totalRunsCount, 1200);
  const animatedActiveModels = useCountUp(activeModelsCount, 1200);
  const animatedPassRate = useCountUp(Math.round(passRateVal), 1200);

  const hasModels = totalModelsCount > 0;
  const recentModels = models.slice(0, 3);

  // Helper for model badge variant based on persisted status
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active':
        return <Badge status="active" type="pass" size="sm">ACTIVE</Badge>;
      case 'building':
        return <Badge status="building" type="review" size="sm">BUILDING</Badge>;
      case 'error':
        return <Badge status="error" type="reject" size="sm">ERROR</Badge>;
      default:
        return <Badge status={status} size="sm">{status.toUpperCase()}</Badge>;
    }
  };

  return (
    <div className="space-y-16 w-full pb-16 font-sans">
      {/* ---------------------------------------------------------------------- */}
      {/* 1. HERO SECTION & DYNAMIC CTAS                                        */}
      {/* ---------------------------------------------------------------------- */}
      <section className="relative overflow-hidden bg-gradient-to-br from-industrial-900 via-industrial-950 to-industrial-900 text-white rounded-3xl p-8 sm:p-12 border border-industrial-800 shadow-xl">
        {/* Subtle Ambient Background Gradients */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-brand-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-pass-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 grid grid-cols-1 lg:grid-cols-12 gap-10 items-center">
          <div className="lg:col-span-7 space-y-6">
            <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-industrial-800/80 border border-industrial-700/80 text-xs font-mono font-semibold text-brand-300">
              <Zap className="w-3.5 h-3.5 text-brand-400 animate-pulse" />
              <span>Industrial Visual Inspection Platform</span>
              {hasModels && (
                <span className="ml-2 text-pass-400 font-bold border-l border-industrial-700 pl-2">
                  • {totalModelsCount} {totalModelsCount === 1 ? 'Model' : 'Models'} Ready
                </span>
              )}
            </div>

            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-white tracking-tight leading-tight">
              Teach the system what <span className="text-pass-400">GOOD</span> looks like.{' '}
              <span className="text-brand-400">Detect what doesn't.</span>
            </h1>

            <p className="text-industrial-300 text-base sm:text-lg leading-relaxed max-w-2xl font-normal">
              Anomaly Detector learns normality representation from defect-free reference images and uses it to identify, localize, and explain visual anomalies in new production images.
            </p>

            {/* Architecture Role Highlight */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
              <div className="p-3.5 rounded-2xl bg-industrial-950/70 border border-industrial-800/80 space-y-1">
                <div className="flex items-center space-x-2 font-mono text-xs font-bold text-brand-300">
                  <Cpu className="w-4 h-4 text-brand-400" />
                  <span>PatchCore Engine</span>
                </div>
                <p className="text-xs text-industrial-400">
                  Anomaly detection & spatial heatmap localization.
                </p>
              </div>

              <div className="p-3.5 rounded-2xl bg-industrial-950/70 border border-industrial-800/80 space-y-1">
                <div className="flex items-center space-x-2 font-mono text-xs font-bold text-indigo-300">
                  <Sparkles className="w-4 h-4 text-indigo-400" />
                  <span>Gemini AI Analysis</span>
                </div>
                <p className="text-xs text-industrial-400">
                  Downstream defect explanation & severity rating.
                </p>
              </div>
            </div>

            {/* Dynamic CTAs */}
            <div className="pt-4 flex flex-wrap items-center gap-4">
              {hasModels ? (
                <>
                  <Button
                    variant="primary"
                    size="lg"
                    onClick={() => navigate('/inspect')}
                    icon={<Play className="w-5 h-5 fill-white" />}
                  >
                    Start Inspection
                  </Button>
                  <Button
                    variant="outline"
                    size="lg"
                    onClick={() => navigate('/models')}
                    icon={<Cpu className="w-5 h-5 text-industrial-300" />}
                    className="border-industrial-700 text-industrial-200 hover:bg-industrial-800 hover:text-white"
                  >
                    View Models ({totalModelsCount})
                  </Button>
                </>
              ) : (
                <>
                  <Button
                    variant="primary"
                    size="lg"
                    onClick={() => navigate('/models/create')}
                    icon={<Plus className="w-5 h-5" />}
                  >
                    Create Your First Model
                  </Button>
                  <Button
                    variant="outline"
                    size="lg"
                    onClick={() => navigate('/dashboard')}
                    icon={<Activity className="w-5 h-5 text-industrial-300" />}
                    className="border-industrial-700 text-industrial-200 hover:bg-industrial-800 hover:text-white"
                  >
                    Explore Dashboard
                  </Button>
                </>
              )}
            </div>
          </div>

          {/* Interactive Dynamic Pipeline Card */}
          <div className="lg:col-span-5">
            <div className="bg-industrial-950 p-6 rounded-2xl border border-industrial-800 space-y-4 shadow-2xl relative">
              <div className="flex items-center justify-between border-b border-industrial-800 pb-3">
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-pass-500 animate-pulse" />
                  <span className="font-mono text-xs font-bold uppercase tracking-wider text-industrial-300">
                    Live Pipeline Visualizer
                  </span>
                </div>
                <span className="text-[10px] font-mono text-industrial-500">Auto Flow</span>
              </div>

              {/* Dynamic Interactive Flow Nodes */}
              <div className="space-y-2.5 text-xs font-mono">
                {[
                  { step: 0, title: '01. GOOD References', desc: 'Defect-free baseline images', color: 'border-pass-500/50 bg-pass-950/20 text-pass-300' },
                  { step: 1, title: '02. PatchCore Extraction', desc: 'Divided local patch features', color: 'border-brand-500/50 bg-brand-950/20 text-brand-300' },
                  { step: 2, title: '03. Memory Bank', desc: 'Normality representation stored', color: 'border-indigo-500/50 bg-indigo-950/20 text-indigo-300' },
                  { step: 3, title: '04. New Image Compare', desc: 'Patch distance scoring', color: 'border-sky-500/50 bg-sky-950/20 text-sky-300' },
                  { step: 4, title: '05. Anomaly Localization', desc: 'Heatmap & BBox spatial grid', color: 'border-amber-500/50 bg-amber-950/20 text-amber-300' },
                  { step: 5, title: '06. Gemini Defect AI', desc: 'Downstream explanation for REJECT', color: 'border-purple-500/50 bg-purple-950/20 text-purple-300' },
                ].map((item) => {
                  const isActive = activePipelineStep === item.step;
                  return (
                    <div
                      key={item.step}
                      onClick={() => setActivePipelineStep(item.step)}
                      className={`p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
                        isActive
                          ? `${item.color} shadow-lg ring-1 ring-current scale-[1.02]`
                          : 'border-industrial-800/80 bg-industrial-900/40 text-industrial-400 hover:bg-industrial-900/80'
                      }`}
                    >
                      <div className="space-y-0.5">
                        <span className="font-bold block text-xs">{item.title}</span>
                        <span className="text-[11px] opacity-80">{item.desc}</span>
                      </div>
                      {isActive && <ChevronRight className="w-4 h-4 animate-bounce-x" />}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------------------- */}
      {/* 2. STEP-BY-STEP "HOW IT WORKS" SECTION                                 */}
      {/* ---------------------------------------------------------------------- */}
      <section className="space-y-8">
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-brand-50 border border-brand-200 text-brand-700 text-xs font-mono font-bold uppercase tracking-wider">
            <Sliders className="w-3.5 h-3.5 text-brand-600" />
            <span>Inspection Workflow</span>
          </div>
          <h2 className="text-3xl font-extrabold text-industrial-900 tracking-tight">
            How Anomaly Detector Works
          </h2>
          <p className="text-industrial-600 text-sm sm:text-base font-semibold">
            From uploading defect-free reference images to AI-assisted defect diagnostics in 6 simple steps.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[
            {
              step: '01',
              title: 'Create a Model',
              desc: 'Define an inspection model dedicated to the product variant you want to inspect (e.g. Screw, PCB, Tile, Textile).',
              icon: Plus,
              accent: 'bg-brand-50 text-brand-600 border-brand-200',
            },
            {
              step: '02',
              title: 'Add GOOD References',
              desc: 'Upload defect-free images. The platform does not require predefined defect categories—it learns what normal looks like.',
              icon: FileCheck,
              accent: 'bg-pass-50 text-pass-600 border-pass-200',
            },
            {
              step: '03',
              title: 'Build Normality Baseline',
              desc: 'PatchCore extracts patch-level visual features and constructs a coreset memory bank representing normal product appearance.',
              icon: Database,
              accent: 'bg-indigo-50 text-indigo-600 border-indigo-200',
            },
            {
              step: '04',
              title: 'Inspect New Images',
              desc: 'Upload new inspection images. PatchCore compares each patch against the learned normality representation.',
              icon: Eye,
              accent: 'bg-sky-50 text-sky-600 border-sky-200',
            },
            {
              step: '05',
              title: 'Localize & Verdict',
              desc: 'If an anomaly exceeds the calibrated threshold, the system provides an anomaly score, spatial heatmap, bounding box, and PASS / REJECT verdict.',
              icon: Target,
              accent: 'bg-amber-50 text-amber-600 border-amber-200',
            },
            {
              step: '06',
              title: 'AI Defect Analysis',
              desc: 'For REJECT images, optionally generate Gemini AI analysis to explain defect type, location, severity, and visual evidence.',
              icon: Sparkles,
              accent: 'bg-purple-50 text-purple-600 border-purple-200',
            },
          ].map((item) => {
            const IconComponent = item.icon;
            return (
              <Card key={item.step} className="p-6 space-y-4 hover:border-brand-300 transition-all hover:shadow-md">
                <div className="flex items-center justify-between">
                  <div className={`p-3 rounded-2xl border ${item.accent}`}>
                    <IconComponent className="w-6 h-6" />
                  </div>
                  <span className="font-mono text-2xl font-black text-industrial-300">
                    {item.step}
                  </span>
                </div>
                <div className="space-y-1.5">
                  <h3 className="text-lg font-extrabold text-industrial-900">{item.title}</h3>
                  <p className="text-xs text-industrial-600 font-semibold leading-relaxed">
                    {item.desc}
                  </p>
                </div>
              </Card>
            );
          })}
        </div>
      </section>

      {/* ---------------------------------------------------------------------- */}
      {/* 3. DYNAMIC APPLICATION STATISTICS & DYNAMIC WORKSPACE SECTION          */}
      {/* ---------------------------------------------------------------------- */}
      <section className="space-y-8">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-industrial-200 pb-4 gap-4">
          <div>
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-industrial-100 text-industrial-700 text-xs font-mono font-bold uppercase tracking-wider mb-2">
              <Activity className="w-3.5 h-3.5 text-industrial-600" />
              <span>Live Application Data</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-extrabold text-industrial-900 tracking-tight">
              Workspace Overview & Live Statistics
            </h2>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate('/dashboard')}
            icon={<ArrowRight className="w-4 h-4" />}
          >
            Open Full Dashboard
          </Button>
        </div>

        {/* Live Dynamic Stats Grid (Retrieved from real backend) */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          <Card className="p-5 space-y-2 border-industrial-200 bg-white">
            <span className="text-[11px] font-mono font-extrabold uppercase tracking-wider text-industrial-500 block">
              Total Models
            </span>
            <div className="flex items-baseline space-x-2">
              <span className="text-3xl font-black font-mono text-industrial-900 tracking-tight">
                {isLoading ? '-' : animatedModels}
              </span>
              <span className="text-xs text-industrial-500 font-semibold">Configured</span>
            </div>
          </Card>

          <Card className="p-5 space-y-2 border-industrial-200 bg-white">
            <span className="text-[11px] font-mono font-extrabold uppercase tracking-wider text-industrial-500 block">
              Images Inspected
            </span>
            <div className="flex items-baseline space-x-2">
              <span className="text-3xl font-black font-mono text-industrial-900 tracking-tight">
                {isLoading ? '-' : animatedInspections}
              </span>
              <span className="text-xs text-industrial-500 font-semibold">Evaluated</span>
            </div>
          </Card>

          <Card className="p-5 space-y-2 border-industrial-200 bg-white">
            <span className="text-[11px] font-mono font-extrabold uppercase tracking-wider text-industrial-500 block">
              Inspection Runs
            </span>
            <div className="flex items-baseline space-x-2">
              <span className="text-3xl font-black font-mono text-industrial-900 tracking-tight">
                {isLoading ? '-' : animatedRuns}
              </span>
              <span className="text-xs text-industrial-500 font-semibold">Batches</span>
            </div>
          </Card>

          <Card className="p-5 space-y-2 border-industrial-200 bg-white">
            <span className="text-[11px] font-mono font-extrabold uppercase tracking-wider text-industrial-500 block">
              Active Models
            </span>
            <div className="flex items-baseline space-x-2">
              <span className="text-3xl font-black font-mono text-brand-600 tracking-tight">
                {isLoading ? '-' : animatedActiveModels}
              </span>
              <span className="text-xs text-industrial-500 font-semibold">In Production</span>
            </div>
          </Card>

          <Card className="p-5 space-y-2 border-industrial-200 bg-white">
            <span className="text-[11px] font-mono font-extrabold uppercase tracking-wider text-industrial-500 block">
              Pass Rate
            </span>
            <div className="flex items-baseline space-x-2">
              <span className="text-3xl font-black font-mono text-pass-600 tracking-tight">
                {isLoading ? '-' : `${animatedPassRate}%`}
              </span>
              <span className="text-xs text-industrial-500 font-semibold">
                ({rejectCountVal} REJECTs)
              </span>
            </div>
          </Card>
        </div>

        {/* Dynamic Workspace Models List (Sorted recent-first with persisted status) */}
        <div className="space-y-4 pt-2">
          <h3 className="text-lg font-extrabold text-industrial-900 flex items-center justify-between">
            <span>Your Inspection Models</span>
            {hasModels && (
              <button
                type="button"
                onClick={() => navigate('/models')}
                className="text-xs font-mono font-bold text-brand-600 hover:text-brand-700 flex items-center space-x-1"
              >
                <span>View All ({totalModelsCount})</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            )}
          </h3>

          {isLoading ? (
            <LoadingSpinner label="Loading workspace models..." size="md" />
          ) : hasModels ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {recentModels.map((m) => (
                <Card
                  key={m.id || m._id}
                  className="p-6 space-y-4 hover:border-brand-400 transition-all hover:shadow-md bg-white cursor-pointer group"
                  onClick={() => navigate(`/models/${m.id || m._id}`)}
                >
                  <div className="flex items-start justify-between">
                    <div className="p-2.5 rounded-xl bg-industrial-100 text-industrial-800 font-mono font-bold text-xs uppercase">
                      <Cpu className="w-5 h-5 text-brand-600" />
                    </div>
                    {/* Persisted Model Status Badge */}
                    {getStatusBadge(m.status)}
                  </div>

                  <div className="space-y-1">
                    <h4 className="text-base font-extrabold text-industrial-900 group-hover:text-brand-600 transition-colors">
                      {m.name}
                    </h4>
                    <p className="text-xs text-industrial-500 line-clamp-2 font-medium">
                      {m.description || 'No description provided.'}
                    </p>
                  </div>

                  <div className="pt-3 border-t border-industrial-100 flex items-center justify-between text-xs font-mono text-industrial-500 font-bold">
                    <span>{m.reference_image_count} Reference Images</span>
                    <span className="text-brand-600 group-hover:translate-x-1 transition-transform flex items-center">
                      Details <ChevronRight className="w-3.5 h-3.5 ml-0.5" />
                    </span>
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            /* Graceful Empty State for workspace */
            <Card className="p-8 text-center space-y-4 border-dashed border-industrial-300 bg-industrial-50/50">
              <div className="w-12 h-12 rounded-full bg-industrial-100 text-industrial-500 flex items-center justify-center mx-auto">
                <Cpu className="w-6 h-6 text-industrial-400" />
              </div>
              <div className="space-y-1 max-w-md mx-auto">
                <h4 className="text-base font-extrabold text-industrial-900">
                  Create your first inspection model
                </h4>
                <p className="text-xs text-industrial-600 font-semibold">
                  No inspection models exist in your workspace yet. Get started by defining a model baseline with defect-free reference images.
                </p>
              </div>
              <Button
                variant="primary"
                size="md"
                onClick={() => navigate('/models/create')}
                icon={<Plus className="w-4 h-4" />}
                className="mx-auto"
              >
                Create Inspection Model
              </Button>
            </Card>
          )}
        </div>
      </section>

      {/* ---------------------------------------------------------------------- */}
      {/* 4. VISUAL ARCHITECTURAL PIPELINE SECTION                               */}
      {/* ---------------------------------------------------------------------- */}
      <section className="bg-industrial-950 text-white p-8 sm:p-12 rounded-3xl border border-industrial-800 space-y-8 shadow-xl">
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-industrial-900 border border-industrial-800 text-brand-400 text-xs font-mono font-bold uppercase tracking-wider">
            <Layers className="w-3.5 h-3.5" />
            <span>Architectural Flow</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            PatchCore & AI Pipeline Architecture
          </h2>
          <p className="text-industrial-400 text-xs sm:text-sm font-semibold">
            How visual features flow from reference images to spatial anomaly heatmaps and optional Gemini AI defect analysis.
          </p>
        </div>

        {/* Visual Pipeline Diagram */}
        <div className="max-w-4xl mx-auto space-y-4 font-mono text-xs">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div className="p-4 rounded-2xl bg-industrial-900 border border-industrial-800 space-y-1">
              <span className="text-[10px] font-bold text-pass-400 block uppercase">Step A</span>
              <span className="font-extrabold text-white text-sm block">GOOD References</span>
              <span className="text-[11px] text-industrial-400">Defect-free training images</span>
            </div>
            <div className="p-4 rounded-2xl bg-industrial-900 border border-industrial-800 space-y-1">
              <span className="text-[10px] font-bold text-brand-400 block uppercase">Step B</span>
              <span className="font-extrabold text-white text-sm block">Feature Extraction</span>
              <span className="text-[11px] text-industrial-400">Local image patch embeddings</span>
            </div>
            <div className="p-4 rounded-2xl bg-industrial-900 border border-industrial-800 space-y-1">
              <span className="text-[10px] font-bold text-indigo-400 block uppercase">Step C</span>
              <span className="font-extrabold text-white text-sm block">Memory Bank</span>
              <span className="text-[11px] text-industrial-400">Coreset normality baseline</span>
            </div>
          </div>

          <div className="flex justify-center text-industrial-600">
            <ArrowRight className="w-6 h-6 transform rotate-90 md:rotate-0" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div className="p-4 rounded-2xl bg-industrial-900 border border-industrial-800 space-y-1">
              <span className="text-[10px] font-bold text-sky-400 block uppercase">Step D</span>
              <span className="font-extrabold text-white text-sm block">New Inspection Image</span>
              <span className="text-[11px] text-industrial-400">Unseen test item</span>
            </div>
            <div className="p-4 rounded-2xl bg-industrial-900 border border-industrial-800 space-y-1">
              <span className="text-[10px] font-bold text-amber-400 block uppercase">Step E</span>
              <span className="font-extrabold text-white text-sm block">Patch-Level Compare</span>
              <span className="text-[11px] text-industrial-400">Nearest-neighbor distance</span>
            </div>
            <div className="p-4 rounded-2xl bg-industrial-900 border border-industrial-800 space-y-1">
              <span className="text-[10px] font-bold text-rose-400 block uppercase">Step F</span>
              <span className="font-extrabold text-white text-sm block">Score & Threshold</span>
              <span className="text-[11px] text-industrial-400">Calibrated PASS vs REJECT</span>
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-industrial-900/90 border border-industrial-800 flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center space-x-3 text-pass-400 font-bold">
              <CheckCircle2 className="w-5 h-5 flex-shrink-0" />
              <span>PASS: Item matches learned normality baseline within threshold limit.</span>
            </div>
            <div className="flex items-center space-x-3 text-rose-400 font-bold border-t md:border-t-0 md:border-l border-industrial-800 pt-3 md:pt-0 md:pl-4">
              <XCircle className="w-5 h-5 flex-shrink-0" />
              <span>REJECT: Heatmap + BBox localized ➔ Optional Gemini AI Defect Analysis</span>
            </div>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------------------- */}
      {/* 5. USE CASES ("EXAMPLE APPLICATIONS")                                  */}
      {/* ---------------------------------------------------------------------- */}
      <section className="space-y-8">
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-industrial-100 text-industrial-700 text-xs sm:text-sm font-mono font-bold uppercase tracking-wider">
            <Factory className="w-4 h-4 text-industrial-600" />
            <span>Domain Versatility</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-industrial-900 tracking-tight">
            Example Inspection Applications
          </h2>
          <p className="text-industrial-600 text-base sm:text-lg font-semibold">
            Anomaly Detector is built for generic visual anomaly detection across diverse industrial sectors.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-6">
          {[
            {
              title: 'Manufacturing QC',
              icon: Factory,
              defects: ['Scratches', 'Dents', 'Surface defects', 'Deformation', 'Contamination'],
            },
            {
              title: 'Electronics / PCB',
              icon: CircuitBoard,
              defects: ['Solder anomalies', 'Missing components', 'Board damage', 'Placement errors'],
            },
            {
              title: 'Textiles',
              icon: Shirt,
              defects: ['Holes', 'Stitching defects', 'Stains', 'Tears', 'Weave abnormalities'],
            },
            {
              title: 'Packaging',
              icon: Package,
              defects: ['Label defects', 'Print issues', 'Package damage', 'Structural anomalies'],
            },
            {
              title: 'Infrastructure',
              icon: Building2,
              defects: ['Cracks', 'Corrosion', 'Spalling', 'Surface degradation'],
            },
          ].map((useCase) => {
            const Icon = useCase.icon;
            return (
              <Card key={useCase.title} className="p-6 space-y-3.5 bg-white border-industrial-200 shadow-xs">
                <div className="p-3 rounded-xl bg-industrial-100 text-industrial-800 w-fit font-bold">
                  <Icon className="w-6 h-6 text-brand-600" />
                </div>
                <h3 className="text-base sm:text-lg font-extrabold text-industrial-900">{useCase.title}</h3>
                <ul className="space-y-2 text-sm text-industrial-700 font-semibold">
                  {useCase.defects.map((d) => (
                    <li key={d} className="flex items-center space-x-2">
                      <span className="w-2 h-2 rounded-full bg-brand-500 flex-shrink-0" />
                      <span>{d}</span>
                    </li>
                  ))}
                </ul>
              </Card>
            );
          })}
        </div>
      </section>

      {/* ---------------------------------------------------------------------- */}
      {/* 6. WHY LEARN NORMALITY? (USER CORRECTION #1)                           */}
      {/* ---------------------------------------------------------------------- */}
      <section className="bg-white p-8 sm:p-10 rounded-3xl border border-industrial-200 shadow-xs space-y-6">
        <div className="max-w-3xl space-y-3">
          <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-brand-50 text-brand-700 text-xs sm:text-sm font-mono font-bold uppercase tracking-wider">
            <HelpCircle className="w-4 h-4 text-brand-600" />
            <span>Core Methodology</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-industrial-900 tracking-tight">
            Why Learn Normality?
          </h2>
          <p className="text-industrial-600 text-base sm:text-lg font-semibold leading-relaxed">
            Traditional supervised defect classification relies on collecting and labeling thousands of examples for every specific defect type. Anomaly Detector instead builds a baseline representation of normal visual appearance from GOOD reference images.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 pt-2">
          <div className="p-5 rounded-2xl bg-industrial-50 border border-industrial-200 space-y-2.5">
            <div className="font-mono text-xs sm:text-sm font-black text-brand-600 uppercase">01. Rare Defects</div>
            <h3 className="text-base sm:text-lg font-extrabold text-industrial-900">Defects are Rare or Unpredictable</h3>
            <p className="text-sm text-industrial-700 font-medium leading-relaxed">
              High-yield manufacturing processes rarely produce defect samples for supervised dataset training.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-industrial-50 border border-industrial-200 space-y-2.5">
            <div className="font-mono text-xs sm:text-sm font-black text-brand-600 uppercase">02. Novel Defects</div>
            <h3 className="text-base sm:text-lg font-extrabold text-industrial-900">Unseen Anomaly Types</h3>
            <p className="text-sm text-industrial-700 font-medium leading-relaxed">
              New or unexpected defect shapes are detected automatically without retraining predefined classifiers.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-industrial-50 border border-industrial-200 space-y-2.5">
            <div className="font-mono text-xs sm:text-sm font-black text-brand-600 uppercase">03. Label Cost</div>
            <h3 className="text-base sm:text-lg font-extrabold text-industrial-900">Zero Labeled Defect Data Needed</h3>
            <p className="text-sm text-industrial-700 font-medium leading-relaxed">
              Eliminates time-consuming pixel-level defect labeling across thousands of training images.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-industrial-50 border border-industrial-200 space-y-2.5">
            <div className="font-mono text-xs sm:text-sm font-black text-brand-600 uppercase">04. Rapid Onboarding</div>
            <h3 className="text-base sm:text-lg font-extrabold text-industrial-900">Fast Model Creation</h3>
            <p className="text-sm text-industrial-700 font-medium leading-relaxed">
              Onboard new product lines in minutes simply by providing a small set of defect-free reference images.
            </p>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------------------- */}
      {/* 7. PATCHCORE TECHNICAL EXPLANATION & GEMINI AI ANALYSIS                */}
      {/* ---------------------------------------------------------------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* PatchCore "Under the Hood" */}
        <Card className="p-7 space-y-4 bg-white border-industrial-200 shadow-xs">
          <div className="flex items-center space-x-3 border-b border-industrial-100 pb-3">
            <div className="p-2.5 rounded-xl bg-industrial-100 text-brand-600">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-xl font-extrabold text-industrial-900">Under the Hood: PatchCore</h3>
              <p className="text-xs sm:text-sm text-industrial-500 font-mono font-bold">Anomaly Detection Engine</p>
            </div>
          </div>
          <ol className="space-y-3 text-sm text-industrial-800 font-medium list-decimal list-inside leading-relaxed">
            <li><strong className="text-industrial-900 font-bold">Local Patch Feature Extraction:</strong> Deep CNN backbones extract multi-scale patch embeddings.</li>
            <li><strong className="text-industrial-900 font-bold">Coreset Memory Bank:</strong> Minimized memory bank represents normal visual variation efficiently.</li>
            <li><strong className="text-industrial-900 font-bold">Distance Comparison:</strong> Test patches are evaluated against memory bank nearest neighbors.</li>
            <li><strong className="text-industrial-900 font-bold">Spatial Localization:</strong> Distance scores yield continuous anomaly heatmaps and bounding boxes.</li>
            <li><strong className="text-industrial-900 font-bold">Threshold Decision:</strong> Calibrated score threshold determines PASS vs REJECT.</li>
          </ol>
        </Card>

        {/* Gemini AI Defect Analysis */}
        <Card className="p-7 space-y-4 bg-white border-industrial-200 shadow-xs">
          <div className="flex items-center space-x-3 border-b border-industrial-100 pb-3">
            <div className="p-2.5 rounded-xl bg-indigo-50 text-indigo-600">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-xl font-extrabold text-industrial-900">AI-Powered Defect Analysis</h3>
              <p className="text-xs sm:text-sm text-industrial-500 font-mono font-bold">Gemini Explanatory Layer</p>
            </div>
          </div>
          <div className="space-y-3.5 text-sm text-industrial-800 font-medium">
            <p>
              PatchCore answers: <strong className="text-industrial-900 font-mono font-bold">"Is this image anomalous and where?"</strong>
            </p>
            <p>
              Gemini helps answer: <strong className="text-indigo-700 font-mono font-bold">"What might the detected anomaly be?"</strong>
            </p>
            <div className="p-4 bg-industrial-50 rounded-xl border border-industrial-200 space-y-1.5 font-mono text-xs sm:text-sm">
              <span className="text-industrial-600 font-bold block">Generated for REJECTed inspections on demand:</span>
              <ul className="list-disc list-inside space-y-1 text-industrial-900 font-medium">
                <li>Defect type & classification proposal</li>
                <li>Localized spatial position description</li>
                <li>Severity rating (Low / Medium / High / Critical)</li>
                <li>Prominence & visual evidence breakdown</li>
              </ul>
            </div>
          </div>
        </Card>
      </div>

      {/* ---------------------------------------------------------------------- */}
      {/* 8. FINAL PRODUCT FLOW & CTA BANNER                                     */}
      {/* ---------------------------------------------------------------------- */}
      <section className="bg-gradient-to-r from-brand-600 via-brand-500 to-amber-600 text-white rounded-3xl p-8 sm:p-12 shadow-xl flex flex-col sm:flex-row items-center justify-between gap-6">
        <div className="space-y-2.5 max-w-xl text-center sm:text-left">
          <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight">
            Ready to start inspecting?
          </h2>
          <p className="text-brand-100 text-sm sm:text-base font-semibold leading-relaxed">
            Create an inspection model using defect-free reference images, build its normality baseline, and start inspecting new products instantly.
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-center sm:justify-end gap-3.5">
          <Button
            variant="ghost"
            size="lg"
            onClick={() => navigate(hasModels ? '/inspect' : '/models/create')}
            icon={<Play className="w-5 h-5 text-brand-600 fill-brand-600 group-hover:!text-white group-hover:!fill-white transition-colors duration-200" />}
            className="group bg-white hover:!bg-black text-industrial-950 hover:!text-white shadow-lg font-extrabold focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-brand-600 border border-white transition-all duration-200"
          >
            {hasModels ? 'Start Inspection' : 'Create Inspection Model'}
          </Button>
          <Button
            variant="ghost"
            size="lg"
            onClick={() => navigate('/dashboard')}
            icon={<Activity className="w-5 h-5 text-white" />}
            className="bg-industrial-950/80 hover:bg-industrial-950 text-white border-2 border-white/70 hover:border-white shadow-lg font-extrabold focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-brand-600"
          >
            View Dashboard
          </Button>
        </div>
      </section>
    </div>
  );
};

export default Home;
