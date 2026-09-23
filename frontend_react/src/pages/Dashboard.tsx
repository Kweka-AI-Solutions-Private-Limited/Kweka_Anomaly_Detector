import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  CheckCircle2,
  XCircle,
  TrendingUp,
  Cpu,
  Layers,
  Play,
  Plus,
  ArrowRight,
  Filter,
  RefreshCw,
  Eye,
  AlertTriangle,
  Activity,
  BarChart3,
  PieChart,
  Clock,
  MessageSquare,
  Info,
  Target,
} from 'lucide-react';
import { getDashboardSummary, DashboardSummaryResponse, DailyTrendItem, AnomalyHotspotAnalysis } from '../api/dashboard';
import { getModels, getModelVersions } from '../api/models';
import { getStorageUrl } from '../api/client';
import { Model, ModelVersion } from '../types';
import { StatCard } from '../components/common/StatCard';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { Button } from '../components/common/Button';
import { LoadingSpinner } from '../components/common/LoadingSpinner';

// ----------------------------------------------------------------------
// Reusable Section Header with Bold Title & Definition/Formula Hover Tooltip
// ----------------------------------------------------------------------
const SectionHeader: React.FC<{
  title: string;
  definition: string;
  formula?: string;
  icon: React.ReactNode;
}> = ({ title, definition, formula, icon }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div className="relative flex items-center justify-between border-b border-industrial-100 pb-3">
      <div
        className="flex items-center space-x-2 cursor-help group"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
      >
        {icon}
        <h2 className="text-sm font-mono font-black uppercase tracking-wider text-industrial-900 group-hover:text-brand-600 transition-colors">
          {title}
        </h2>
        <Info className="w-4 h-4 text-industrial-400 group-hover:text-brand-600 transition-colors" />
      </div>

      {showTooltip && (
        <div className="absolute z-30 left-0 top-8 w-80 bg-industrial-900 text-white text-xs font-sans p-4 rounded-xl shadow-2xl border border-industrial-700 pointer-events-none space-y-2.5 animate-in fade-in duration-200">
          <div className="font-mono font-bold text-brand-300 uppercase tracking-wider text-xs border-b border-industrial-800 pb-1">
            {title} Component Info
          </div>
          <div>
            <span className="font-mono font-bold text-industrial-400 text-xs block uppercase mb-0.5">Definition:</span>
            <span className="text-industrial-200 text-xs leading-relaxed block font-medium">{definition}</span>
          </div>
          {formula && (
            <div className="bg-industrial-950 p-2.5 rounded-lg border border-industrial-800 font-mono text-xs text-amber-300 font-medium">
              <span className="font-bold text-industrial-400 text-xs block uppercase mb-0.5">Formula:</span>
              <code>{formula}</code>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ----------------------------------------------------------------------
// 1. SVG Time-Series Line Chart Component (Inspection Activity)
// ----------------------------------------------------------------------
const LineChart: React.FC<{ trend: DailyTrendItem[] }> = ({ trend }) => {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!trend || trend.length === 0) {
    return (
      <div className="h-56 flex flex-col items-center justify-center text-industrial-400 font-mono text-xs border border-dashed border-industrial-200 rounded-xl bg-industrial-50/50">
        <Activity className="w-8 h-8 mb-2 opacity-40 text-industrial-400" />
        <span>No inspection trend data available for this period.</span>
      </div>
    );
  }

  const items = trend.length === 1 ? [
    { date: 'Prev', total: 0, pass: 0, reject: 0 },
    trend[0]
  ] : trend;

  const width = 600;
  const height = 200;
  const paddingX = 45;
  const paddingY = 25;
  const chartW = width - paddingX * 2;
  const chartH = height - paddingY * 2;

  const maxVal = Math.max(1, ...items.map((i) => Math.max(i.total, i.pass, i.reject)));

  const points = items.map((item, idx) => {
    const x = paddingX + (idx / (items.length - 1)) * chartW;
    const yTotal = height - paddingY - (item.total / maxVal) * chartH;
    const yPass = height - paddingY - (item.pass / maxVal) * chartH;
    const yReject = height - paddingY - (item.reject / maxVal) * chartH;
    return { x, yTotal, yPass, yReject, item, idx };
  });

  const passPathD = points.reduce((acc, p, idx) => `${acc} ${idx === 0 ? 'M' : 'L'} ${p.x} ${p.yPass}`, '');
  const passAreaD = `${passPathD} L ${points[points.length - 1].x} ${height - paddingY} L ${points[0].x} ${height - paddingY} Z`;

  const rejectPathD = points.reduce((acc, p, idx) => `${acc} ${idx === 0 ? 'M' : 'L'} ${p.x} ${p.yReject}`, '');
  const rejectAreaD = `${rejectPathD} L ${points[points.length - 1].x} ${height - paddingY} L ${points[0].x} ${height - paddingY} Z`;

  return (
    <div className="relative w-full overflow-hidden">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible select-none">
        <defs>
          <linearGradient id="passGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
          </linearGradient>
          <linearGradient id="rejectGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#f43f5e" stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const y = height - paddingY - ratio * chartH;
          const val = Math.round(ratio * maxVal);
          return (
            <g key={ratio}>
              <line x1={paddingX} y1={y} x2={width - paddingX} y2={y} stroke="#e2e8f0" strokeDasharray="3 3" strokeWidth="1" />
              <text x={paddingX - 8} y={y + 4} textAnchor="end" className="text-[9px] fill-industrial-400 font-mono">
                {val}
              </text>
            </g>
          );
        })}

        <path d={passAreaD} fill="url(#passGrad)" />
        <path d={rejectAreaD} fill="url(#rejectGrad)" />

        <path d={passPathD} fill="none" stroke="#10b981" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d={rejectPathD} fill="none" stroke="#f43f5e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />

        {points.map((p) => {
          // Format "2026-09-22" -> "09/22" to prevent date string overflow
          let displayDate = p.item.date;
          if (displayDate && displayDate.includes('-')) {
            const parts = displayDate.split('-');
            if (parts.length === 3) {
              displayDate = `${parts[1]}/${parts[2]}`;
            }
          }

          // If there are many data points, show every Nth date label plus first & last
          const totalPoints = points.length;
          const step = totalPoints > 10 ? 3 : totalPoints > 6 ? 2 : 1;
          const showText = p.idx === 0 || p.idx === totalPoints - 1 || p.idx % step === 0;

          return (
            <g key={p.idx} onMouseEnter={() => setHoveredIdx(p.idx)} onMouseLeave={() => setHoveredIdx(null)}>
              {showText && (
                <text
                  x={p.x}
                  y={height - 4}
                  textAnchor="middle"
                  className="text-[10px] sm:text-xs fill-industrial-600 font-mono font-black"
                >
                  {displayDate}
                </text>
              )}
              <circle
                cx={p.x}
                cy={p.yPass}
                r={hoveredIdx === p.idx ? 5.5 : 4}
                fill="#10b981"
                stroke="#ffffff"
                strokeWidth="2"
                className="transition-all cursor-pointer"
              />
              <circle
                cx={p.x}
                cy={p.yReject}
                r={hoveredIdx === p.idx ? 5.5 : 4}
                fill="#f43f5e"
                stroke="#ffffff"
                strokeWidth="2"
                className="transition-all cursor-pointer"
              />
            </g>
          );
        })}
      </svg>

      {hoveredIdx !== null && points[hoveredIdx] && (
        <div
          className="absolute z-10 bg-industrial-900 text-white text-xs font-mono p-2.5 rounded-lg shadow-xl pointer-events-none transform -translate-x-1/2 -translate-y-full border border-industrial-700"
          style={{
            left: `${(points[hoveredIdx].x / width) * 100}%`,
            top: `${(points[hoveredIdx].yPass / height) * 100 - 10}%`,
          }}
        >
          <div className="font-bold border-b border-industrial-700 pb-1 mb-1 text-industrial-300">
            {points[hoveredIdx].item.date}
          </div>
          <div className="text-emerald-400 font-semibold">PASS: {points[hoveredIdx].item.pass}</div>
          <div className="text-rose-400 font-semibold">REJECT: {points[hoveredIdx].item.reject}</div>
          <div className="text-industrial-400 text-[10px] mt-0.5">Total: {points[hoveredIdx].item.total}</div>
        </div>
      )}
    </div>
  );
};

// ----------------------------------------------------------------------
// 2. SVG Donut Chart Component (PASS vs REJECT Part-to-Whole)
// ----------------------------------------------------------------------
const DonutChart: React.FC<{ passCount: number; rejectCount: number; passPct: number; rejectPct: number }> = ({
  passCount,
  rejectCount,
  passPct,
  rejectPct,
}) => {
  const total = passCount + rejectCount;

  if (total === 0) {
    return (
      <div className="h-48 flex flex-col items-center justify-center text-industrial-400 font-mono text-xs border border-dashed border-industrial-200 rounded-xl bg-industrial-50/50">
        <PieChart className="w-8 h-8 mb-2 opacity-40 text-industrial-400" />
        <span>No inspection data for this filter scope.</span>
      </div>
    );
  }

  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const passStroke = (passPct / 100) * circumference;
  const rejectStroke = (rejectPct / 100) * circumference;

  return (
    <div className="flex flex-col sm:flex-row items-center justify-around gap-6">
      <div className="relative w-44 h-44 flex items-center justify-center">
        <svg viewBox="0 0 160 160" className="w-full h-full transform -rotate-90">
          <circle cx="80" cy="80" r={radius} fill="none" stroke="#e2e8f0" strokeWidth="18" />
          <circle
            cx="80"
            cy="80"
            r={radius}
            fill="none"
            stroke="#10b981"
            strokeWidth="18"
            strokeDasharray={`${passStroke} ${circumference}`}
            strokeDashoffset="0"
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />
          <circle
            cx="80"
            cy="80"
            r={radius}
            fill="none"
            stroke="#f43f5e"
            strokeWidth="18"
            strokeDasharray={`${rejectStroke} ${circumference}`}
            strokeDashoffset={`-${passStroke}`}
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          <span className="text-2xl font-black font-mono text-industrial-900 tracking-tight">{passPct}%</span>
          <span className="text-xs font-mono font-black uppercase tracking-wider text-industrial-600">PASS RATE</span>
        </div>
      </div>

      <div className="space-y-3 font-mono text-xs w-full sm:w-auto">
        <div className="p-3 bg-pass-50/60 border border-pass-200/80 rounded-xl flex items-center justify-between space-x-4">
          <div className="flex items-center space-x-2">
            <span className="w-3 h-3 rounded-full bg-pass-500" />
            <span className="font-black text-pass-900">PASS</span>
          </div>
          <div className="text-right">
            <span className="font-black text-pass-900">{passCount}</span>
            <span className="text-pass-700 text-xs ml-1.5 font-bold">({passPct}%)</span>
          </div>
        </div>

        <div className="p-3 bg-reject-50/60 border border-reject-200/80 rounded-xl flex items-center justify-between space-x-4">
          <div className="flex items-center space-x-2">
            <span className="w-3 h-3 rounded-full bg-reject-500" />
            <span className="font-black text-reject-900">REJECT</span>
          </div>
          <div className="text-right">
            <span className="font-black text-reject-900">{rejectCount}</span>
            <span className="text-reject-700 text-xs ml-1.5 font-bold">({rejectPct}%)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// ----------------------------------------------------------------------
// 3. 5x5 Spatial Grid Anomaly Hotspot Component
// ----------------------------------------------------------------------
const HotspotGrid: React.FC<{ hotspotData?: AnomalyHotspotAnalysis }> = ({ hotspotData }) => {
  if (!hotspotData || hotspotData.total_anomalous_inspections === 0 || hotspotData.inspections_with_bbox === 0) {
    return (
      <div className="h-56 flex flex-col items-center justify-center text-industrial-400 font-mono text-xs border border-dashed border-industrial-200 rounded-xl bg-industrial-50/50 p-4 space-y-1">
        <AlertTriangle className="w-8 h-8 mb-1 opacity-40 text-industrial-400" />
        <span className="font-bold text-industrial-700">No Anomaly Hotspot Data</span>
        <span className="text-industrial-400 text-center text-[11px] max-w-xs">
          No anomalous inspections with spatial localization bounding boxes found for this scope.
        </span>
      </div>
    );
  }

  const { matrix, max_cell_count, total_anomalous_inspections, inspections_with_bbox } = hotspotData;

  let peakRow = -1;
  let peakCol = -1;
  if (max_cell_count > 0) {
    for (let r = 0; r < 5; r++) {
      for (let c = 0; c < 5; c++) {
        if (matrix[r][c] === max_cell_count) {
          peakRow = r + 1;
          peakCol = c + 1;
          break;
        }
      }
      if (peakRow !== -1) break;
    }
  }

  return (
    <div className="space-y-3">
      {/* 5x5 Spatial Matrix Container */}
      <div className="bg-industrial-950 p-3.5 rounded-2xl border border-industrial-800 space-y-2.5">
        <div className="flex items-center justify-between text-[10px] font-mono text-industrial-400 font-bold uppercase tracking-wider px-1">
          <span>Spatial Grid (5x5 BBox Overlap)</span>
          <span className="text-amber-400 font-black">
            {max_cell_count > 0 ? `Peak Zone: Row ${peakRow}, Col ${peakCol}` : 'No Overlap'}
          </span>
        </div>

        <div className="grid grid-cols-5 gap-1.5 aspect-square max-w-[240px] mx-auto">
          {matrix.map((row, rIdx) =>
            row.map((count, cIdx) => {
              const intensity = max_cell_count > 0 ? count / max_cell_count : 0;
              let bgStyle = 'bg-industrial-900 text-industrial-500 border-industrial-800';
              if (count > 0) {
                if (intensity >= 0.75) {
                  bgStyle = 'bg-rose-600 text-white border-rose-500 shadow-sm shadow-rose-900/50 font-black';
                } else if (intensity >= 0.4) {
                  bgStyle = 'bg-rose-500/80 text-white border-rose-400 font-bold';
                } else if (intensity >= 0.2) {
                  bgStyle = 'bg-amber-500/70 text-white border-amber-400 font-semibold';
                } else {
                  bgStyle = 'bg-amber-500/30 text-amber-200 border-amber-500/40';
                }
              }

              return (
                <div
                  key={`cell-${rIdx}-${cIdx}`}
                  title={`Row ${rIdx + 1}, Col ${cIdx + 1}: ${count} bbox area overlap(s)`}
                  className={`flex flex-col items-center justify-center p-1 rounded-lg border text-xs font-mono transition-all hover:scale-105 cursor-pointer ${bgStyle}`}
                >
                  <span className="text-[8px] opacity-65 leading-none">R{rIdx + 1}C{cIdx + 1}</span>
                  <span className="text-[11px] font-bold leading-tight mt-0.5">{count}</span>
                </div>
              );
            })
          )}
        </div>

        {/* Legend */}
        <div className="flex items-center justify-between pt-2 border-t border-industrial-800 text-[10px] font-mono text-industrial-400">
          <div className="flex items-center space-x-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-industrial-900 border border-industrial-800 inline-block" />
            <span>0</span>
            <span className="w-2.5 h-2.5 rounded-sm bg-amber-500/30 border border-amber-500/40 inline-block ml-1" />
            <span className="w-2.5 h-2.5 rounded-sm bg-amber-500/70 inline-block" />
            <span className="w-2.5 h-2.5 rounded-sm bg-rose-500 inline-block" />
            <span className="w-2.5 h-2.5 rounded-sm bg-rose-600 inline-block" />
            <span>Max ({max_cell_count})</span>
          </div>
          <span className="font-bold text-industrial-300">Area Overlap</span>
        </div>
      </div>

      {/* Summary Footer */}
      <div className="grid grid-cols-2 gap-2 text-xs font-mono">
        <div className="p-2.5 bg-industrial-50 border border-industrial-200 rounded-xl text-center">
          <span className="block text-[10px] font-bold text-industrial-500 uppercase">REJECT Inspections</span>
          <span className="text-sm font-black text-industrial-900">{total_anomalous_inspections}</span>
        </div>
        <div className="p-2.5 bg-industrial-50 border border-industrial-200 rounded-xl text-center">
          <span className="block text-[10px] font-bold text-industrial-500 uppercase">With PatchCore BBox</span>
          <span className="text-sm font-black text-brand-700">{inspections_with_bbox}</span>
        </div>
      </div>
    </div>
  );
};

// ----------------------------------------------------------------------
// 3. Main Dashboard Page Component
// ----------------------------------------------------------------------
export const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  const [dashboardData, setDashboardData] = useState<DashboardSummaryResponse | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [selectedVersion, setSelectedVersion] = useState<string>('all');
  const [availableVersions, setAvailableVersions] = useState<ModelVersion[]>([]);
  const [isLoadingVersions, setIsLoadingVersions] = useState<boolean>(false);
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');

  useEffect(() => {
    async function loadModels() {
      try {
        const mList = await getModels();
        setModels(mList);
      } catch (err) {
        console.error('Failed to load models list:', err);
      }
    }
    loadModels();
  }, []);

  // Fetch available versions whenever a specific model is selected
  useEffect(() => {
    if (selectedModel === 'all') {
      setAvailableVersions([]);
      setSelectedVersion('all');
      return;
    }

    async function loadVersions() {
      try {
        setIsLoadingVersions(true);
        setSelectedVersion('all');
        const vList = await getModelVersions(selectedModel);
        const sorted = [...vList].sort((a, b) => b.version_number - a.version_number);
        setAvailableVersions(sorted);
      } catch (err) {
        console.error('Failed to load model versions:', err);
        setAvailableVersions([]);
      } finally {
        setIsLoadingVersions(false);
      }
    }
    loadVersions();
  }, [selectedModel]);

  const fetchDashboardMetrics = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await getDashboardSummary({
        model_id: selectedModel,
        model_version_id: selectedVersion,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
      });
      setDashboardData(data);
    } catch (err: any) {
      console.error('Error fetching dashboard summary:', err);
      setError('Failed to load dashboard aggregation metrics.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardMetrics();
  }, [selectedModel, selectedVersion, startDate, endDate]);

  const handleResetFilters = () => {
    setSelectedModel('all');
    setSelectedVersion('all');
    setStartDate('');
    setEndDate('');
  };

  if (isLoading && !dashboardData) {
    return <LoadingSpinner label="Loading dashboard operational metrics..." size="lg" />;
  }

  const kpi = dashboardData?.kpi || {
    total_inspections: 0,
    pass_count: 0,
    reject_count: 0,
    error_count: 0,
    pass_rate: 0.0,
    total_runs: 0,
    active_models: 0,
  };

  const passVsReject = dashboardData?.pass_vs_reject || {
    pass_count: 0,
    reject_count: 0,
    pass_percentage: 0.0,
    reject_percentage: 0.0,
  };

  const trend = dashboardData?.inspection_trend || [];
  const defects = dashboardData?.defect_distribution || [];
  const hotspotData = dashboardData?.anomaly_hotspot_analysis;
  const severity = dashboardData?.severity_distribution || { Low: 0, Medium: 0, High: 0, Critical: 0 };
  const modelDist = dashboardData?.model_distribution || [];
  const runSum = dashboardData?.run_summary || { total_runs: 0, completed_runs: 0, failed_runs: 0, recent_runs: [] };
  const feedbackSum = dashboardData?.feedback_summary || { total_feedback: 0, categories: { correct: 0, false_positive: 0, false_negative: 0, wrong_defect_type: 0, wrong_location: 0, wrong_severity: 0 } };
  const recentInspections = dashboardData?.recent_inspections || [];

  const maxDefectCount = Math.max(1, ...defects.map((d) => d.count));
  const totalSeverityCount = Math.max(1, Object.values(severity).reduce((a, b) => a + b, 0));
  const maxModelCount = Math.max(1, ...modelDist.map((m) => m.count));

  return (
    <div className="space-y-8 w-full pb-10">
      {/* HEADER SECTION & SCOPE FILTERS */}
      <div className="bg-white p-6 sm:p-7 rounded-2xl border border-industrial-200 shadow-xs space-y-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div>
            <h1 className="text-2xl sm:text-3xl font-black text-industrial-900 tracking-tight">
              Inspection Quality Dashboard
            </h1>
            <p className="text-sm text-industrial-600 mt-1 font-semibold">
              Industrial visual-inspection analytics, operational KPIs, and defect distributions.
            </p>
          </div>

          <div className="flex items-center space-x-3 self-start md:self-auto">
            <Button
              variant="outline"
              size="md"
              onClick={() => navigate('/models/create')}
              icon={<Plus className="w-4 h-4 text-brand-600" />}
            >
              Create Model
            </Button>
            <Button
              variant="primary"
              size="md"
              onClick={() => navigate('/inspect')}
              icon={<Play className="w-4 h-4 fill-white" />}
            >
              Run Inspection
            </Button>
          </div>
        </div>

        {/* Scope Controls */}
        <div className="pt-4 border-t border-industrial-100 flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center space-x-2 text-xs font-black text-industrial-900 font-mono uppercase mr-1">
              <Filter className="w-3.5 h-3.5 text-brand-600" />
              <span>Scope:</span>
            </div>

            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="px-3.5 py-1.5 text-xs border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none bg-white font-black text-industrial-900 shadow-2xs"
            >
              <option value="all">All Models ({models.length})</option>
              {models.map((m) => (
                <option key={m.id || m._id} value={m.id || m._id}>
                  {m.name}
                </option>
              ))}
            </select>

            <select
              value={selectedVersion}
              onChange={(e) => setSelectedVersion(e.target.value)}
              disabled={selectedModel === 'all' || isLoadingVersions}
              title={selectedModel === 'all' ? 'Select a specific model to filter by version' : 'Filter by model version'}
              className={`px-3.5 py-1.5 text-xs border rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none font-black shadow-2xs transition-colors ${
                selectedModel === 'all'
                  ? 'bg-industrial-100 text-industrial-400 border-industrial-200 cursor-not-allowed'
                  : 'bg-white text-industrial-900 border-industrial-300 cursor-pointer'
              }`}
            >
              <option value="all">
                {selectedModel === 'all'
                  ? 'All Versions'
                  : isLoadingVersions
                  ? 'Loading versions...'
                  : `All Versions (${availableVersions.length})`}
              </option>
              {availableVersions.map((v) => (
                <option key={v.id || v._id} value={v.id || v._id}>
                  V{v.version_number} ({v.status})
                </option>
              ))}
            </select>

            <div className="flex items-center space-x-1.5">
              <span className="text-[11px] font-mono text-industrial-700 font-black">From:</span>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="px-2.5 py-1 text-xs border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none bg-white font-black text-industrial-900 shadow-2xs"
              />
            </div>

            <div className="flex items-center space-x-1.5">
              <span className="text-[11px] font-mono text-industrial-700 font-black">To:</span>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="px-2.5 py-1 text-xs border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none bg-white font-black text-industrial-900 shadow-2xs"
              />
            </div>

            {(selectedModel !== 'all' || selectedVersion !== 'all' || startDate || endDate) && (
              <button
                type="button"
                onClick={handleResetFilters}
                className="inline-flex items-center space-x-1 px-2.5 py-1 text-xs font-black border border-industrial-300 rounded-xl hover:bg-industrial-50 text-industrial-700 transition-colors font-mono"
              >
                <RefreshCw className="w-3 h-3" />
                <span>Reset</span>
              </button>
            )}
          </div>

          <div className="text-xs font-mono text-industrial-700 font-black">
            Scope Records: <strong className="text-industrial-900 font-black">{kpi.total_inspections}</strong>
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-reject-50 border border-reject-200 text-reject-700 text-xs rounded-xl font-bold">
          {error}
        </div>
      )}

      {/* ---------------------------------------------------------------------- */}
      {/* TIER 1: KPI ROW WITH BOLD TITLES & DEFINITION/FORMULA TOOLTIPS          */}
      {/* ---------------------------------------------------------------------- */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        <StatCard
          title="Total Inspections"
          value={kpi.total_inspections}
          change="PASS + REJECT + ERR"
          isPositive={true}
          icon={Layers}
          definition="Total count of evaluated visual inspection images for selected scope."
          formula="Total = PASS + REJECT + ERROR"
        />
        <StatCard
          title="PASS"
          value={kpi.pass_count}
          change={`${passVsReject.pass_percentage}% of total`}
          isPositive={true}
          icon={CheckCircle2}
          definition="Number of inspected items meeting quality criteria without defects."
          formula="PASS Count"
        />
        <StatCard
          title="REJECT"
          value={kpi.reject_count}
          change={`${passVsReject.reject_percentage}% of total`}
          isPositive={kpi.reject_count === 0}
          icon={XCircle}
          definition="Number of inspected items flagged with visual anomalies or defects."
          formula="REJECT Count"
        />
        <StatCard
          title="Pass Rate"
          value={`${kpi.pass_rate}%`}
          change="PASS / (PASS + REJECT)"
          isPositive={kpi.pass_rate >= 85}
          icon={TrendingUp}
          definition="Percentage of valid inspections that passed quality criteria."
          formula="Pass Rate = (PASS / (PASS + REJECT)) × 100%"
        />
        <StatCard
          title="Total Runs"
          value={kpi.total_runs}
          change={`${runSum.completed_runs} completed`}
          isPositive={true}
          icon={Play}
          definition="Number of batch inspection runs executed for selected scope."
          formula="Total Runs = Completed Runs + Failed Runs"
        />
        <StatCard
          title="Active Models"
          value={kpi.active_models}
          change={`${models.length} registered`}
          isPositive={true}
          icon={Cpu}
          definition="Number of registered AI models evaluating parts in selected scope."
          formula="Count of Active Models"
        />
      </div>

      {/* ---------------------------------------------------------------------- */}
      {/* TIER 2: HERO VISUALIZATION (INSPECTION ACTIVITY)                      */}
      {/* ---------------------------------------------------------------------- */}
      <Card className="p-6 space-y-4 bg-white border-industrial-200 shadow-xs">
        <SectionHeader
          title="Inspection Activity (Time-Series Volume)"
          definition="Daily volume of inspections processed over time split by PASS and REJECT verdicts."
          formula="Daily Total = Daily PASS + Daily REJECT"
          icon={<Activity className="w-4 h-4 text-brand-600" />}
        />
        <LineChart trend={trend} />
      </Card>

      {/* ---------------------------------------------------------------------- */}
      {/* TIER 2 & 3: OUTCOME + ANOMALY HOTSPOT ANALYSIS                         */}
      {/* ---------------------------------------------------------------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6 space-y-4 bg-white border-industrial-200 shadow-xs">
          <SectionHeader
            title="PASS vs REJECT Outcome Ratio"
            definition="Proportion of passed versus rejected items represented as a part-to-whole donut chart."
            formula="PASS % = (PASS / Total Valid) × 100%, REJECT % = (REJECT / Total Valid) × 100%"
            icon={<PieChart className="w-4 h-4 text-brand-600" />}
          />
          <DonutChart
            passCount={passVsReject.pass_count}
            rejectCount={passVsReject.reject_count}
            passPct={passVsReject.pass_percentage}
            rejectPct={passVsReject.reject_percentage}
          />
        </Card>

        <Card className="p-6 space-y-4 bg-white border-industrial-200 shadow-xs">
          <SectionHeader
            title="Anomaly Hotspot Analysis"
            definition="Spatial 5x5 grid heatmap of anomaly localization bounding box occurrences derived from PatchCore."
            formula="Area Overlap across 5x5 Spatial Grid"
            icon={<Target className="w-4 h-4 text-brand-600" />}
          />
          <HotspotGrid hotspotData={hotspotData} />
        </Card>
      </div>

      {/* ---------------------------------------------------------------------- */}
      {/* TIER 3: SEVERITY + MODEL DISTRIBUTION                                  */}
      {/* ---------------------------------------------------------------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6 space-y-4 bg-white border-industrial-200 shadow-xs">
          <SectionHeader
            title="Severity Distribution"
            definition="Breakdown of visual defects by severity rating (Low, Medium, High, Critical)."
            formula="Severity % = (Severity Count / Total Defects) × 100%"
            icon={<BarChart3 className="w-4 h-4 text-brand-600" />}
          />

          <div className="space-y-3 pt-1">
            {[
              { label: 'Low', count: severity.Low || 0, color: 'bg-industrial-400', textColor: 'text-industrial-800' },
              { label: 'Medium', count: severity.Medium || 0, color: 'bg-brand-500', textColor: 'text-brand-800' },
              { label: 'High', count: severity.High || 0, color: 'bg-amber-500', textColor: 'text-amber-800' },
              { label: 'Critical', count: severity.Critical || 0, color: 'bg-reject-500', textColor: 'text-reject-800' },
            ].map((sev) => {
              const pct = Math.round((sev.count / totalSeverityCount) * 100);
              return (
                <div key={sev.label} className="space-y-1.5 font-mono text-xs">
                  <div className="flex items-center justify-between">
                    <span className={`font-black ${sev.textColor}`}>{sev.label}</span>
                    <span className="font-black text-industrial-900">{sev.count} ({pct}%)</span>
                  </div>
                  <div className="w-full bg-industrial-100 h-2.5 rounded-full overflow-hidden">
                    <div
                      className={`${sev.color} h-full rounded-full transition-all duration-500`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        <Card className="p-6 space-y-4 bg-white border-industrial-200 shadow-xs">
          <SectionHeader
            title="Inspections by Model"
            definition="Inspection volume breakdown evaluated across registered AI models."
            formula="Inspection Count per Model"
            icon={<BarChart3 className="w-4 h-4 text-brand-600" />}
          />

          {modelDist.length === 0 ? (
            <div className="h-44 flex flex-col items-center justify-center text-industrial-400 font-mono text-xs border border-dashed border-industrial-200 rounded-xl bg-industrial-50/50">
              <span>No model inspection data available.</span>
            </div>
          ) : (
            <div className="space-y-3 pt-1">
              {modelDist.map((mItem) => {
                const barWidth = Math.round((mItem.count / maxModelCount) * 100);
                return (
                  <div key={mItem.model_name} className="space-y-1.5 font-mono text-xs">
                    <div className="flex items-center justify-between text-industrial-900">
                      <span className="font-black truncate max-w-[220px]">{mItem.model_name}</span>
                      <span className="font-black text-brand-700">{mItem.count} Inspections</span>
                    </div>
                    <div className="w-full bg-industrial-100 h-2.5 rounded-full overflow-hidden">
                      <div
                        className="bg-brand-500 h-full rounded-full transition-all duration-500"
                        style={{ width: `${barWidth}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </Card>
      </div>

      {/* ---------------------------------------------------------------------- */}
      {/* TIER 4: RECENT RUNS & FEEDBACK SUMMARY                                 */}
      {/* ---------------------------------------------------------------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6 space-y-4 bg-white border-industrial-200 shadow-xs">
          <SectionHeader
            title="Recent Inspection Runs"
            definition="Log of the most recent batch inspection runs with model version and image counts."
            formula="Latest Batch Run Log"
            icon={<Clock className="w-4 h-4 text-brand-600" />}
          />

          {runSum.recent_runs.length === 0 ? (
            <div className="p-6 text-center text-industrial-400 font-mono text-xs border border-dashed border-industrial-200 rounded-xl bg-industrial-50/50">
              No recent runs found for this scope.
            </div>
          ) : (
            <div className="space-y-2">
              {runSum.recent_runs.map((r) => (
                <div
                  key={r.id}
                  className="flex items-center justify-between text-xs font-mono p-3 bg-industrial-50/70 rounded-xl border border-industrial-200/80 hover:bg-industrial-100/50 transition-colors"
                >
                  <div className="space-y-0.5">
                    <span className="font-black text-brand-700 block">
                      Run #{r.run_number}
                    </span>
                    <span className="text-industrial-700 text-[11px] font-sans font-bold">
                      {r.model_name} (V{r.version_number})
                    </span>
                  </div>
                  <div className="text-right space-y-0.5">
                    <span className="px-2 py-0.5 bg-brand-50 text-brand-700 font-black rounded-md border border-brand-200 text-[11px]">
                      {r.total_images} Images
                    </span>
                    <span className="block text-[10px] text-industrial-500 uppercase font-black">
                      {r.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card className="p-6 space-y-4 bg-white border-industrial-200 shadow-xs">
          <SectionHeader
            title="Scoped Feedback Summary"
            definition="User-submitted feedback accuracy metrics for inspections in selected scope."
            formula="Feedback Category Totals"
            icon={<MessageSquare className="w-4 h-4 text-brand-600" />}
          />

          {feedbackSum.total_feedback === 0 ? (
            <div className="h-44 flex flex-col items-center justify-center text-industrial-400 font-mono text-xs border border-dashed border-industrial-200 rounded-xl bg-industrial-50/50">
              <MessageSquare className="w-8 h-8 mb-2 opacity-40 text-industrial-400" />
              <span>No feedback submitted yet for this filter scope.</span>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <div className="p-3 bg-pass-50/60 border border-pass-200 rounded-xl text-center">
                <span className="block text-[10px] font-mono font-black text-pass-800 uppercase">Correct</span>
                <span className="text-base font-mono font-black text-pass-900">{feedbackSum.categories.correct}</span>
              </div>
              <div className="p-3 bg-amber-50/60 border border-amber-200 rounded-xl text-center">
                <span className="block text-[10px] font-mono font-black text-amber-800 uppercase">False Pos</span>
                <span className="text-base font-mono font-black text-amber-900">{feedbackSum.categories.false_positive}</span>
              </div>
              <div className="p-3 bg-reject-50/60 border border-reject-200 rounded-xl text-center">
                <span className="block text-[10px] font-mono font-black text-reject-800 uppercase">False Neg</span>
                <span className="text-base font-mono font-black text-reject-900">{feedbackSum.categories.false_negative}</span>
              </div>
              <div className="p-3 bg-industrial-50/80 border border-industrial-200 rounded-xl text-center">
                <span className="block text-[10px] font-mono font-black text-industrial-700 uppercase">Wrong Defect</span>
                <span className="text-base font-mono font-black text-industrial-900">{feedbackSum.categories.wrong_defect_type}</span>
              </div>
              <div className="p-3 bg-industrial-50/80 border border-industrial-200 rounded-xl text-center">
                <span className="block text-[10px] font-mono font-black text-industrial-700 uppercase">Wrong Loc</span>
                <span className="text-base font-mono font-black text-industrial-900">{feedbackSum.categories.wrong_location}</span>
              </div>
              <div className="p-3 bg-industrial-50/80 border border-industrial-200 rounded-xl text-center">
                <span className="block text-[10px] font-mono font-black text-industrial-700 uppercase">Wrong Sev</span>
                <span className="text-base font-mono font-black text-industrial-900">{feedbackSum.categories.wrong_severity}</span>
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* ---------------------------------------------------------------------- */}
      {/* TIER 4: RECENT INSPECTION LOGS TABLE                                   */}
      {/* ---------------------------------------------------------------------- */}
      <Card className="p-6 space-y-4 bg-white border-industrial-200 shadow-xs">
        <div className="flex items-center justify-between border-b border-industrial-100 pb-3">
          <SectionHeader
            title="Recent Inspection Logs"
            definition="Detailed audit table of the latest 10 individual inspection records."
            formula="Latest Inspection Records Feed"
            icon={<Layers className="w-4 h-4 text-brand-600" />}
          />
          <button
            onClick={() => navigate('/history')}
            className="text-xs font-black text-brand-600 hover:text-brand-700 flex items-center space-x-1.5 transition-colors font-mono"
          >
            <span>View Full History</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>

        {recentInspections.length === 0 ? (
          <div className="p-10 text-center text-industrial-400 font-mono text-xs border border-dashed border-industrial-200 rounded-xl bg-industrial-50/50">
            No inspection logs found for this filter scope.
          </div>
        ) : (
          <div className="overflow-x-auto border border-industrial-200 rounded-xl shadow-2xs">
            <table className="w-full text-left font-mono text-xs sm:text-sm">
              <thead className="bg-industrial-50 text-industrial-800 uppercase border-b border-industrial-200 font-black text-xs sm:text-sm">
                <tr>
                  <th className="py-3.5 px-4">Thumbnail</th>
                  <th className="py-3.5 px-4">Filename</th>
                  <th className="py-3.5 px-4 font-black text-industrial-900">Model</th>
                  <th className="py-3.5 px-4">Version</th>
                  <th className="py-3.5 px-4">Run</th>
                  <th className="py-3.5 px-4">Verdict</th>
                  <th className="py-3.5 px-4">Severity</th>
                  <th className="py-3.5 px-4">Timestamp</th>
                  <th className="py-3.5 px-4 text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-industrial-200 bg-white">
                {recentInspections.map((item) => {
                  const storageUrl = getStorageUrl(item.storage_uri);
                  return (
                    <tr
                      key={item.id}
                      onClick={() => navigate(`/inspections/${item.id}`)}
                      className="hover:bg-brand-50/40 cursor-pointer transition-colors"
                    >
                      <td className="py-3.5 px-4">
                        {storageUrl ? (
                          <img
                            src={storageUrl}
                            alt="Sample"
                            className="w-10 h-10 object-cover rounded-lg border border-industrial-200 bg-industrial-100"
                          />
                        ) : (
                          <div className="w-10 h-10 rounded-lg bg-industrial-100 border border-industrial-200 flex items-center justify-center text-xs text-industrial-400 font-sans font-medium">
                            N/A
                          </div>
                        )}
                      </td>
                      <td className="py-3.5 px-4 font-black text-industrial-900 truncate max-w-[140px] text-xs sm:text-sm">
                        {item.filename}
                      </td>
                      <td className="py-3.5 px-4 font-sans font-black text-industrial-900 truncate max-w-[140px] text-xs sm:text-sm">
                        {item.model_name}
                      </td>
                      <td className="py-3.5 px-4 text-industrial-800 font-bold text-xs sm:text-sm">
                        {item.model_version_number != null ? `V${item.model_version_number}` : '—'}
                      </td>
                      <td className="py-3.5 px-4 text-xs sm:text-sm">
                        {item.run_number != null ? (
                          <span className="font-black text-brand-700">Run #{item.run_number}</span>
                        ) : (
                          <span className="text-industrial-400 italic font-sans text-xs sm:text-sm">Standalone</span>
                        )}
                      </td>
                      <td className="py-3.5 px-4">
                        <Badge status={item.verdict === 'PASS' ? 'normal' : 'anomalous'} size="sm" />
                      </td>
                      <td className="py-3.5 px-4 font-sans font-black text-industrial-800 text-xs sm:text-sm">
                        {item.severity}
                      </td>
                      <td className="py-3.5 px-4 text-industrial-700 text-xs sm:text-sm whitespace-nowrap font-bold">
                        {item.created_at ? new Date(item.created_at).toLocaleString() : 'Just now'}
                      </td>
                      <td className="py-3.5 px-4 text-center">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            navigate(`/inspections/${item.id}`);
                          }}
                          className="text-brand-600 hover:text-brand-700 hover:underline inline-flex items-center space-x-1 font-sans font-black text-xs sm:text-sm"
                        >
                          <Eye className="w-4 h-4" />
                          <span>View</span>
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
};
