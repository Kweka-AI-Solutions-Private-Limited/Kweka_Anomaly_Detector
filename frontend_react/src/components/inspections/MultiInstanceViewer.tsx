import React, { useState } from 'react';
import {
  CheckCircle2, XCircle, AlertTriangle, Layers, Eye, EyeOff, Sparkles,
  Search, ShieldAlert, Cpu, Activity, ExternalLink
} from 'lucide-react';
import { InspectionResultResponse, InstanceResult, OverallPrediction } from '../../types';
import { VLMDefectAnalysisCard } from './VLMDefectAnalysisCard';
import { LoadingSpinner } from '../common/LoadingSpinner';

interface MultiInstanceViewerProps {
  result: InspectionResultResponse;
  onTriggerInstanceVlm?: (instanceId: number) => Promise<void>;
  isVlmLoading?: boolean;
}

export const MultiInstanceViewer: React.FC<MultiInstanceViewerProps> = ({
  result,
  onTriggerInstanceVlm,
  isVlmLoading = false,
}) => {
  if (!result) {
    return (
      <div className="p-8 bg-slate-900/80 border border-slate-800 rounded-xl text-center text-slate-400 space-y-3">
        <LoadingSpinner label="Loading multi-instance inspection details..." size="md" />
      </div>
    );
  }

  const instances = result?.instances || (result as any)?.result?.instances || [];
  const overall = result?.overall_prediction || (result as any)?.result?.overall_prediction;

  const [selectedInstanceId, setSelectedInstanceId] = useState<number | null>(
    instances && instances.length > 0 ? instances[0].instance_id : null
  );
  const [showCompositeHeatmap, setShowCompositeHeatmap] = useState<boolean>(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(0.65);
  const [hoveredInstanceId, setHoveredInstanceId] = useState<number | null>(null);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number }>({ width: 0, height: 0 });

  const selectedInstance = instances.find((i: InstanceResult) => i.instance_id === selectedInstanceId) || instances[0];

  const getStatusBadge = (status: string) => {
    const uppercaseStatus = status?.toUpperCase() || 'UNKNOWN';
    if (uppercaseStatus === 'PASS' || uppercaseStatus === 'NORMAL') {
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          PASS
        </span>
      );
    }
    if (uppercaseStatus === 'REJECT' || uppercaseStatus === 'ANOMALOUS') {
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">
          <XCircle className="w-3.5 h-3.5 text-rose-400" />
          REJECT
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
        <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
        {uppercaseStatus}
      </span>
    );
  };

  const getBboxBorderColor = (inst: InstanceResult, isSelected: boolean, isHovered: boolean) => {
    const status = inst.prediction?.status?.toUpperCase();
    if (isSelected || isHovered) {
      return 'border-cyan-400 ring-2 ring-cyan-400/50 z-30 shadow-lg shadow-cyan-500/20';
    }
    if (status === 'PASS' || status === 'NORMAL') {
      return 'border-emerald-500/80 bg-emerald-500/10 hover:border-emerald-400';
    }
    if (status === 'REJECT' || status === 'ANOMALOUS') {
      return 'border-rose-500/90 bg-rose-500/15 hover:border-rose-400 animate-pulse-subtle';
    }
    return 'border-amber-500/80 bg-amber-500/10 hover:border-amber-400';
  };

  // Convert storage_uri to same-origin relative media URL
  const formatMediaUrl = (uri?: string | null) => {
    if (!uri) return '';
    if (uri.startsWith('http://') || uri.startsWith('https://')) return uri;
    const cleanPath = uri.startsWith('/') ? uri.slice(1) : uri;
    if (!cleanPath.startsWith('storage/')) {
      return `/storage/${cleanPath}`;
    }
    return `/${cleanPath}`;
  };

  const storageUri = result?.storage_uri || (result as any)?.result?.storage_uri;
  const compositeHeatmapUri = result?.composite_heatmap_uri || (result as any)?.result?.composite_heatmap_uri;

  const imgWidth = imageDimensions.width || (result as any)?.processing_stats?.image_width || (result as any)?.result?.processing_stats?.image_width || 1000;
  const imgHeight = imageDimensions.height || (result as any)?.processing_stats?.image_height || (result as any)?.result?.processing_stats?.image_height || 1000;

  return (
    <div className="space-y-6">
      {/* Overall Inspection Summary Header */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 backdrop-blur-sm">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
              <Layers className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-bold text-slate-100">
                  Multi-Product Inspection Result
                </h3>
                {overall && getStatusBadge(overall.status)}
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                {overall?.message || `Detected ${instances.length} product instance(s) via Gemini Instance Localization. PatchCore evaluated each crop.`}
              </p>
            </div>
          </div>

          {overall && (
            <div className="flex items-center gap-4 text-xs">
              <div className="bg-slate-950/60 px-3 py-2 rounded-lg border border-slate-800/80">
                <span className="text-slate-400 block">Total Instances</span>
                <span className="font-semibold text-slate-200 text-sm">{overall.total_instances}</span>
              </div>
              <div className="bg-emerald-500/10 px-3 py-2 rounded-lg border border-emerald-500/20">
                <span className="text-emerald-400 block">Passed</span>
                <span className="font-semibold text-emerald-300 text-sm">{overall.pass_count}</span>
              </div>
              <div className="bg-rose-500/10 px-3 py-2 rounded-lg border border-rose-500/20">
                <span className="text-rose-400 block font-medium">Rejected</span>
                <span className="font-semibold text-rose-300 text-sm">{overall.reject_count}</span>
              </div>
              {overall.max_anomaly_score !== undefined && (
                <div className="bg-slate-950/60 px-3 py-2 rounded-lg border border-slate-800/80">
                  <span className="text-slate-400 block">Peak Anomaly Score</span>
                  <span className="font-semibold text-amber-300 text-sm">{overall.max_anomaly_score?.toFixed(1)}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Main Grid: Left = Original Image with Interactive Bounding Boxes, Right = Selected Instance Detail */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (7 cols): Full Image Overlay Viewer */}
        <div className="lg:col-span-7 bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Search className="w-4 h-4 text-slate-400" />
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Original Inspection Surface (Gemini Instance Localization)
              </span>
            </div>

            {compositeHeatmapUri && (
              <button
                onClick={() => setShowCompositeHeatmap(!showCompositeHeatmap)}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  showCompositeHeatmap
                    ? 'bg-indigo-600/30 text-indigo-300 border border-indigo-500/40'
                    : 'bg-slate-800 text-slate-400 border border-slate-700 hover:text-slate-200'
                }`}
              >
                {showCompositeHeatmap ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                Composite Heatmap
              </button>
            )}
          </div>

          {/* Image & Bounding Box Container */}
          <div className="relative w-full aspect-square bg-slate-950 rounded-lg overflow-hidden border border-slate-800/80 flex items-center justify-center group">
            {/* Original Base Image */}
            <img
              src={formatMediaUrl(storageUri)}
              alt="Original Multi-Instance Inspection"
              className="w-full h-full object-contain"
              onLoad={(e) => {
                const img = e.currentTarget;
                if (img.naturalWidth && img.naturalHeight) {
                  setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
                }
              }}
            />

            {/* Composite Heatmap Overlay */}
            {showCompositeHeatmap && compositeHeatmapUri && (
              <img
                src={formatMediaUrl(compositeHeatmapUri)}
                alt="Composite Anomaly Heatmap"
                className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-screen transition-opacity"
                style={{ opacity: heatmapOpacity }}
              />
            )}

            {/* Interactive Bounding Box Overlays */}
            <div className="absolute inset-0 w-full h-full pointer-events-none">
              {instances.map((inst: InstanceResult) => {
                const isSelected = inst.instance_id === selectedInstanceId;
                const isHovered = inst.instance_id === hoveredInstanceId;
                const borderStyle = getBboxBorderColor(inst, isSelected, isHovered);

                const bbox = inst.bbox || { x: 0, y: 0, width: 0, height: 0 };
                const leftPct = (bbox.x / imgWidth) * 100;
                const topPct = (bbox.y / imgHeight) * 100;
                const widthPct = (bbox.width / imgWidth) * 100;
                const heightPct = (bbox.height / imgHeight) * 100;

                return (
                  <div
                    key={inst.instance_id}
                    onMouseEnter={() => setHoveredInstanceId(inst.instance_id)}
                    onMouseLeave={() => setHoveredInstanceId(null)}
                    onClick={() => setSelectedInstanceId(inst.instance_id)}
                    style={{
                      left: `${leftPct}%`,
                      top: `${topPct}%`,
                      width: `${widthPct}%`,
                      height: `${heightPct}%`,
                    }}
                    className={`absolute pointer-events-auto cursor-pointer border-2 transition-all rounded-md flex items-start p-1 ${borderStyle}`}
                  >
                    <span className={`px-2 py-0.5 rounded text-xs font-black shadow ${
                      inst.prediction?.status?.toUpperCase() === 'PASS' || inst.prediction?.status?.toUpperCase() === 'NORMAL'
                        ? 'bg-emerald-500 text-slate-950'
                        : 'bg-rose-500 text-white'
                    }`}>
                      #{inst.instance_id} {inst.prediction?.status?.toUpperCase() || 'ROIs'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-3 flex items-center justify-between text-xs sm:text-sm text-slate-300 font-medium px-1">
            <span>Click any box or instance card below to inspect ROI</span>
            {compositeHeatmapUri && (
              <div className="flex items-center gap-2">
                <span>Opacity:</span>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={heatmapOpacity}
                  onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                  className="w-24 accent-indigo-500 cursor-pointer"
                />
              </div>
            )}
          </div>
        </div>

        {/* Right Column (5 cols): Selected Instance ROI Detail & Gemini VLM Card */}
        <div className="lg:col-span-5 bg-slate-900/80 border border-slate-800 rounded-xl p-5 flex flex-col space-y-4">
          {selectedInstance ? (
            <>
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <span className="px-2.5 py-1 bg-indigo-500/10 text-indigo-400 font-bold text-xs sm:text-sm rounded-md border border-indigo-500/20">
                    Instance #{selectedInstance.instance_id}
                  </span>
                  {selectedInstance.prediction && getStatusBadge(selectedInstance.prediction.status)}
                </div>
                <span className="text-xs sm:text-sm text-slate-300 font-medium">
                  Confidence: {((selectedInstance.detection_confidence || 0.95) * 100).toFixed(0)}%
                </span>
              </div>

              {/* Crop Image + Heatmap Side by Side */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800 flex flex-col items-center">
                  <span className="text-xs text-slate-300 mb-1 font-semibold">Instance Crop</span>
                  <div className="w-full aspect-square bg-slate-900 rounded overflow-hidden flex items-center justify-center">
                    <img
                      src={formatMediaUrl(selectedInstance.crop_storage_uri)}
                      alt={`Instance ${selectedInstance.instance_id} Crop`}
                      className="w-full h-full object-contain"
                    />
                  </div>
                </div>

                <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800 flex flex-col items-center">
                  <span className="text-xs text-slate-300 mb-1 font-semibold">Anomaly Heatmap</span>
                  <div className="w-full aspect-square bg-slate-900 rounded overflow-hidden flex items-center justify-center">
                    {selectedInstance.localization?.heatmap_uri ? (
                      <img
                        src={formatMediaUrl(selectedInstance.localization.heatmap_uri)}
                        alt={`Instance ${selectedInstance.instance_id} Heatmap`}
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <div className="text-xs text-slate-400 p-2 text-center font-medium">No Anomaly Heatmap</div>
                    )}
                  </div>
                </div>
              </div>

              {/* Score & Threshold Metrics */}
              {selectedInstance.prediction && (
                <div className="bg-slate-950/80 p-3.5 rounded-lg border border-slate-800/80 space-y-2.5 text-xs sm:text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400 font-medium">Anomaly Score:</span>
                    <span className="font-bold text-slate-100 font-mono">
                      {selectedInstance.prediction.anomaly_score != null
                        ? selectedInstance.prediction.anomaly_score.toFixed(2)
                        : 'N/A (Pure Gemini Engine)'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400 font-medium">Threshold:</span>
                    <span className="text-slate-300 font-mono">
                      {selectedInstance.prediction.threshold != null
                        ? selectedInstance.prediction.threshold.toFixed(2)
                        : 'N/A — Not applicable to Gemini'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between pt-1 border-t border-slate-800/60">
                    <span className="text-slate-400 font-medium">VLM Confidence:</span>
                    <span className="font-bold text-cyan-400 font-mono">
                      {selectedInstance.detection_confidence != null
                        ? `${(selectedInstance.detection_confidence * 100).toFixed(0)}%`
                        : 'N/A'}
                    </span>
                  </div>
                  {/* Progress bar only rendered when numeric PatchCore score exists */}
                  {selectedInstance.prediction.anomaly_score != null && selectedInstance.prediction.threshold != null && (
                    <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${
                          selectedInstance.prediction.status?.toUpperCase() === 'REJECT' || selectedInstance.prediction.status?.toUpperCase() === 'ANOMALOUS'
                            ? 'bg-rose-500'
                            : 'bg-emerald-500'
                        }`}
                        style={{
                          width: `${Math.min(
                            100,
                            (selectedInstance.prediction.anomaly_score / (selectedInstance.prediction.threshold * 1.5)) * 100
                          )}%`,
                        }}
                      />
                    </div>
                  )}
                </div>
              )}

              {/* Instance Bounding Box Coordinates */}
              <div className="text-xs text-slate-300 bg-slate-950/70 p-3 rounded-lg border border-slate-800/80 font-mono space-y-1 font-medium">
                <div>
                  Original ROI: x={selectedInstance?.bbox?.x ?? 0}, y={selectedInstance?.bbox?.y ?? 0}, w={selectedInstance?.bbox?.width ?? 0}, h={selectedInstance?.bbox?.height ?? 0}
                </div>
                {selectedInstance.localization?.bbox && (
                  <div className="text-amber-400 font-semibold">
                    Mapped Anomaly Box: x={selectedInstance.localization.bbox.x}, y={selectedInstance.localization.bbox.y}, w={selectedInstance.localization.bbox.width}, h={selectedInstance.localization.bbox.height}
                  </div>
                )}
              </div>

              {/* Instance-Level Gemini VLM Trigger Section */}
              <div className="pt-2">
                {selectedInstance.prediction?.status?.toUpperCase() === 'REJECT' || selectedInstance.prediction?.status?.toUpperCase() === 'ANOMALOUS' ? (
                  selectedInstance.vlm_analysis ? (
                    <VLMDefectAnalysisCard
                      isPass={false}
                      vlmAnalysis={selectedInstance.vlm_analysis}
                      onGenerate={() => onTriggerInstanceVlm && onTriggerInstanceVlm(selectedInstance.instance_id)}
                      isGenerating={isVlmLoading}
                    />
                  ) : (
                    <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl text-center space-y-3">
                      <div className="inline-flex p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
                        <Sparkles className="w-5 h-5" />
                      </div>
                      <h4 className="text-xs sm:text-sm font-bold text-slate-200">
                        AI Defect Analysis for Instance #{selectedInstance.instance_id}
                      </h4>
                      <p className="text-xs text-slate-300 font-medium leading-relaxed">
                        Generate multimodal Gemini insights on defect classification and severity for this crop.
                      </p>
                      <button
                        onClick={() => onTriggerInstanceVlm && onTriggerInstanceVlm(selectedInstance.instance_id)}
                        disabled={isVlmLoading}
                        className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-bold text-xs sm:text-sm rounded-lg transition-all shadow-md shadow-indigo-600/20 disabled:opacity-50"
                      >
                        <Sparkles className="w-4 h-4" />
                        {isVlmLoading ? 'Generating AI Analysis...' : 'Generate AI Defect Analysis'}
                      </button>
                    </div>
                  )
                ) : (
                  <div className="text-center p-3 bg-slate-950/40 rounded-lg border border-slate-800/40 text-xs text-slate-500">
                    Gemini AI analysis skipped for healthy (PASS) instances.
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="text-center py-12 text-slate-500 text-xs">
              Select an instance from the drawer below to view crop details.
            </div>
          )}
        </div>
      </div>

      {/* Bottom Instance Drawer / Grid Selector */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
        <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3">
          Detected Product Instances ({instances.length})
        </h4>

        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-3">
          {instances.map((inst: InstanceResult) => {
            const isSelected = inst.instance_id === selectedInstanceId;
            return (
              <button
                key={inst.instance_id}
                onClick={() => setSelectedInstanceId(inst.instance_id)}
                onMouseEnter={() => setHoveredInstanceId(inst.instance_id)}
                onMouseLeave={() => setHoveredInstanceId(null)}
                className={`p-2.5 rounded-lg border text-left transition-all flex flex-col space-y-2 ${
                  isSelected
                    ? 'bg-slate-800/90 border-cyan-500/80 shadow-md ring-1 ring-cyan-500/40'
                    : 'bg-slate-950/60 border-slate-800 hover:bg-slate-800/40 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-extrabold text-slate-200">#{inst.instance_id}</span>
                  {inst.prediction && getStatusBadge(inst.prediction.status)}
                </div>

                <div className="w-full aspect-square bg-slate-900 rounded overflow-hidden border border-slate-800/80 flex items-center justify-center">
                  <img
                    src={formatMediaUrl(inst.crop_storage_uri)}
                    alt={`Instance ${inst.instance_id}`}
                    className="w-full h-full object-contain"
                  />
                </div>

                {inst.prediction && (
                  <div className="text-[11px] text-slate-400 font-mono text-center">
                    Score: <span className="font-semibold text-slate-200">
                      {inst.prediction.anomaly_score != null ? inst.prediction.anomaly_score.toFixed(1) : 'N/A'}
                    </span>
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default MultiInstanceViewer;
