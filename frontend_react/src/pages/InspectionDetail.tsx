import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
  ArrowLeft,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Send,
  Layers,
  MessageSquare,
  ChevronRight,
  PlayCircle
} from 'lucide-react';
import { getInspection, retryVlmAnalysis, retryInstanceVlmAnalysis } from '../api/inspections';
import { submitFeedback, submitExpandedFeedback, getInspectionFeedback } from '../api/feedback';
import { getModel, getModelVersions } from '../api/models';
import { getInspectionRun } from '../api/runs';
import { getStorageUrl } from '../api/client';
import { InspectionResultResponse, Feedback, Model, InspectionRun, ModelVersion } from '../types';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { Button } from '../components/common/Button';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { VLMDefectAnalysisCard } from '../components/inspections/VLMDefectAnalysisCard';
import { MultiInstanceViewer } from '../components/inspections/MultiInstanceViewer';
import { Check } from 'lucide-react';

export const InspectionDetail: React.FC = () => {
  const { inspectionId } = useParams<{ inspectionId: string }>();
  const navigate = useNavigate();

  const [inspection, setInspection] = useState<InspectionResultResponse | null>(null);
  const [model, setModel] = useState<Model | null>(null);
  const [run, setRun] = useState<InspectionRun | null>(null);
  const [version, setVersion] = useState<ModelVersion | null>(null);

  const [feedbackList, setFeedbackList] = useState<Feedback[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // VLM Retry State
  const [isRetryingVlm, setIsRetryingVlm] = useState<boolean>(false);

  // Expanded Feedback form state
  const [detectionFeedback, setDetectionFeedback] = useState<'correct' | 'false_positive' | 'false_negative'>('correct');
  const [vlmFeedbackCategories, setVlmFeedbackCategories] = useState<('correct' | 'wrong_defect_type' | 'wrong_location' | 'wrong_severity')[]>(['correct']);
  const [correctedDefectType, setCorrectedDefectType] = useState<string>('');
  const [correctedLocation, setCorrectedLocation] = useState<string>('');
  const [correctedSeverity, setCorrectedSeverity] = useState<string>('');
  const [feedbackComment, setFeedbackComment] = useState<string>('');
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState<boolean>(false);
  const [submitSuccess, setSubmitSuccess] = useState<boolean>(false);

  const toggleVlmCategory = (category: 'correct' | 'wrong_defect_type' | 'wrong_location' | 'wrong_severity') => {
    if (category === 'correct') {
      setVlmFeedbackCategories(['correct']);
      return;
    }
    setVlmFeedbackCategories((prev) => {
      const filtered = prev.filter((c) => c !== 'correct');
      if (filtered.includes(category)) {
        const next = filtered.filter((c) => c !== category);
        return next.length === 0 ? ['correct'] : next;
      } else {
        return [...filtered, category];
      }
    });
  };

  const handleFeedbackSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inspectionId) return;

    try {
      setIsSubmittingFeedback(true);
      setSubmitSuccess(false);

      const newFb = await submitExpandedFeedback({
        inspectionId,
        detection_feedback: detectionFeedback,
        vlm_feedback_categories: vlmFeedbackCategories,
        corrected_defect_type: correctedDefectType || null,
        corrected_location: correctedLocation || null,
        corrected_severity: correctedSeverity || null,
        comment: feedbackComment || null,
      });

      setFeedbackList((prev) => [newFb, ...prev]);
      setSubmitSuccess(true);
      setTimeout(() => setSubmitSuccess(false), 4000);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to submit operator feedback.');
    } finally {
      setIsSubmittingFeedback(false);
    }
  };

  const handleGenerateVlm = async (force?: boolean) => {
    if (!inspectionId) return;
    try {
      setIsRetryingVlm(true);
      const updated = await retryVlmAnalysis(inspectionId, force);
      setInspection(updated);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to generate Gemini AI analysis.');
    } finally {
      setIsRetryingVlm(false);
    }
  };

  useEffect(() => {
    async function loadData() {
      if (!inspectionId) return;
      try {
        setIsLoading(true);
        setError(null);

        const [resInspection, resFeedback] = await Promise.all([
          getInspection(inspectionId),
          getInspectionFeedback(inspectionId).catch(() => []),
        ]);
        setInspection(resInspection);
        setFeedbackList(resFeedback);

        // Fetch contextual hierarchy info (Model, Version, Run)
        const rawInsp = resInspection as any;
        const modelId = resInspection.model_id || rawInsp.model_id;
        const versionId = resInspection.model_version_id || rawInsp.model_version_id;
        const runId = rawInsp.run_id;

        if (modelId) {
          getModel(modelId).then(setModel).catch(() => null);
        }

        if (modelId && versionId) {
          getModelVersions(modelId).then(vList => {
            const match = vList.find(v => (v.id || v._id) === versionId);
            if (match) setVersion(match);
          }).catch(() => null);
        }

        if (runId) {
          getInspectionRun(runId).then(setRun).catch(() => null);
        }
      } catch (err: any) {
        setError('Failed to load inspection details.');
      } finally {
        setIsLoading(false);
      }
    }
    loadData();
  }, [inspectionId]);



  if (isLoading) {
    return <LoadingSpinner label="Loading inspection report detail..." size="lg" />;
  }

  if (error || !inspection) {
    return (
      <div className="p-6 bg-white rounded-lg border border-industrial-200 text-center space-y-4">
        <AlertTriangle className="w-8 h-8 text-reject-600 mx-auto" />
        <h3 className="text-base font-bold text-industrial-900">Inspection Not Found</h3>
        <p className="text-xs text-industrial-500">{error || 'The requested inspection ID does not exist.'}</p>
        <Button variant="primary" onClick={() => navigate('/history')}>
          Return to History
        </Button>
      </div>
    );
  }

  const pred = inspection.prediction || (inspection as any).result?.prediction;
  const loc = inspection.localization || (inspection as any).result?.localization;
  const storageUri = inspection.storage_uri || (inspection as any).input?.storage_uri;
  const filename = inspection.filename || (inspection as any).input?.filename || 'sample.png';
  const inspId = inspection.inspection_id || (inspection as any).id || (inspection as any)._id;

  const isAnomalous = pred?.status === 'anomalous' || pred?.status === 'REJECT';
  const sampleUrl = getStorageUrl(storageUri);
  const heatmapUrl = getStorageUrl(loc?.heatmap_uri);

  return (
    <div className="space-y-6">
      {/* Contextual Breadcrumb Navigation */}
      <nav className="flex items-center space-x-2 text-sm font-medium text-industrial-500 flex-wrap">
        <Link to="/models" className="hover:text-brand-600 transition-colors">
          Models
        </Link>
        {model && (
          <>
            <ChevronRight className="w-4 h-4 text-industrial-400" />
            <Link to={`/models/${model.id || model._id}`} className="hover:text-brand-600 font-semibold text-industrial-800">
              {model.name}
            </Link>
          </>
        )}
        {version && (
          <>
            <ChevronRight className="w-4 h-4 text-industrial-400" />
            <span className="text-industrial-700">Version #{version.version_number}</span>
          </>
        )}
        {run ? (
          <>
            <ChevronRight className="w-4 h-4 text-industrial-400" />
            <span className="text-brand-700 font-bold">Run #{run.run_number}</span>
          </>
        ) : (
          <>
            <ChevronRight className="w-4 h-4 text-industrial-400" />
            <span className="text-industrial-500 italic">Standalone</span>
          </>
        )}
        <ChevronRight className="w-4 h-4 text-industrial-400" />
        <span className="text-industrial-900 font-bold truncate max-w-[200px]">{filename}</span>
      </nav>

      {/* Top Header Card */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 bg-white p-5 rounded-2xl border border-industrial-200 shadow-sm">
        <div className="flex items-center space-x-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              if (model) {
                navigate(`/models/${model.id || model._id}`);
              } else {
                navigate(-1);
              }
            }}
            icon={<ArrowLeft className="w-4 h-4" />}
          >
            {run ? `Back to Run #${run.run_number}` : 'Back'}
          </Button>
          <div className="h-6 w-px bg-industrial-200" />
          <div>
            <h1 className="text-lg font-bold text-industrial-900 font-mono flex items-center space-x-2">
              <span>Inspection Record</span>
              <span className="text-xs text-industrial-400 font-normal">({inspId})</span>
            </h1>
            <p className="text-xs text-industrial-500 font-mono mt-0.5">
              File: <strong className="text-industrial-800">{filename}</strong>
              {run && <span className="ml-3 text-brand-700 font-bold">● Run #{run.run_number}</span>}
              {version && <span className="ml-2 text-industrial-600">● Version #{version.version_number}</span>}
            </p>
          </div>
        </div>

        <Badge status={pred?.status || 'normal'} size="lg" />
      </div>

      {/* Main Inspection Visualization Grid */}
      {inspection.inspection_mode === 'multi_instance' ||
      (inspection as any).result?.inspection_mode === 'multi_instance' ||
      (inspection.instances && inspection.instances.length > 0) ||
      ((inspection as any).result?.instances && (inspection as any).result?.instances.length > 0) ? (
        <Card className="p-6 bg-white border-industrial-200 shadow-sm">
          <MultiInstanceViewer
            result={inspection}
            onTriggerInstanceVlm={async (instanceId) => {
              const updated = await retryInstanceVlmAnalysis(inspId, instanceId);
              setInspection(updated);
            }}
          />
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Side: Images Viewport (7 cols) */}
          <Card className="lg:col-span-7 p-6 space-y-4 bg-white border-industrial-200">
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-industrial-600">
              Visual Inspection Data
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Original Sample */}
              <div className="space-y-2">
                <span className="text-xs font-mono font-semibold text-industrial-700">Original Test Image</span>
                <div className="bg-industrial-900 p-2 rounded-xl border border-industrial-300 h-64 flex items-center justify-center">
                {sampleUrl ? (
                  <img src={sampleUrl} alt="Sample" className="max-h-full max-w-full object-contain rounded-lg" />
                ) : (
                  <span className="text-xs text-industrial-500 font-mono">No Image</span>
                )}
              </div>
            </div>

            {/* PatchCore Heatmap */}
            <div className="space-y-2">
              <span className="text-xs font-mono font-semibold text-industrial-700">PatchCore Heatmap</span>
              <div className="bg-industrial-900 p-2 rounded-xl border border-industrial-300 h-64 flex items-center justify-center relative">
                {heatmapUrl ? (
                  <img src={heatmapUrl} alt="Heatmap" className="max-h-full max-w-full object-contain rounded-lg" />
                ) : (
                  <span className="text-xs text-industrial-500 font-mono">No Heatmap Generated</span>
                )}

                {/* Bounding box */}
                {loc?.bbox && (
                  <svg
                    className="absolute inset-0 w-full h-full pointer-events-none"
                    viewBox="0 0 100 100"
                    preserveAspectRatio="none"
                  >
                    <rect
                      x={`${loc.bbox.x}`}
                      y={`${loc.bbox.y}`}
                      width={`${loc.bbox.width}`}
                      height={`${loc.bbox.height}`}
                      fill="none"
                      stroke="#EF4444"
                      strokeWidth="2.5"
                    />
                  </svg>
                )}
              </div>
            </div>
          </div>
        </Card>

        {/* Right Side: Metrics & Operator Feedback (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Result Card */}
          <Card className="p-6 space-y-4 bg-white border-industrial-200">
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-industrial-600">
              Prediction Metrics
            </h2>

            {isAnomalous ? (
              <div className="p-4 bg-reject-50 border border-reject-200 rounded-xl text-center space-y-1">
                <XCircle className="w-8 h-8 text-reject-600 mx-auto" />
                <h3 className="text-lg font-extrabold text-reject-700 font-mono">REJECT</h3>
                <p className="text-xs text-reject-600 font-medium">Anomalous Feature Spikes Detected</p>
              </div>
            ) : (
              <div className="p-4 bg-pass-50 border border-pass-200 rounded-xl text-center space-y-1">
                <CheckCircle2 className="w-8 h-8 text-pass-600 mx-auto" />
                <h3 className="text-lg font-extrabold text-pass-700 font-mono">PASS</h3>
                <p className="text-xs text-pass-600 font-medium">Product Feature Distance Normal</p>
              </div>
            )}

            <div className="grid grid-cols-2 gap-3 font-mono text-xs">
              <div className="p-3 bg-industrial-50 rounded-xl border border-industrial-200">
                <span className="text-[10px] text-industrial-500 uppercase">Score</span>
                <p className="text-base font-bold text-industrial-900 mt-0.5">
                  {pred?.anomaly_score != null ? pred.anomaly_score.toFixed(2) : 'N/A'}
                </p>
              </div>
              <div className="p-3 bg-industrial-50 rounded-xl border border-industrial-200">
                <span className="text-[10px] text-industrial-500 uppercase">Threshold</span>
                <p className="text-base font-bold text-industrial-700 mt-0.5">
                  {pred?.threshold != null ? pred.threshold.toFixed(2) : '27.00'}
                </p>
              </div>
            </div>

            <div className="bg-industrial-50 p-3.5 rounded-xl border border-industrial-200 space-y-2 font-mono text-xs">
              <div className="flex justify-between">
                <span className="text-industrial-500">Severity:</span>
                <span className="font-semibold capitalize text-industrial-900">
                  {pred?.severity || 'none'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-industrial-500">Processing Latency:</span>
                <span className="font-semibold text-industrial-900">{inspection.processing_time_ms ?? 0} ms</span>
              </div>
            </div>

            {/* Bounding box readout */}
            <div className="bg-industrial-50 p-3.5 rounded-xl border border-industrial-200 space-y-1 font-mono text-xs">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold text-industrial-500 uppercase tracking-wider">
                  DETECTED REGION (PatchCore localization)
                </span>
              </div>
              {loc?.bbox ? (
                <div className="grid grid-cols-4 gap-1 text-[11px] text-industrial-800 pt-1">
                  <div><span className="text-industrial-400">X:</span> {loc.bbox.x}</div>
                  <div><span className="text-industrial-400">Y:</span> {loc.bbox.y}</div>
                  <div><span className="text-industrial-400">W:</span> {loc.bbox.width}</div>
                  <div><span className="text-industrial-400">H:</span> {loc.bbox.height}</div>
                </div>
              ) : (
                <p className="text-[11px] text-industrial-400 italic pt-1">No localized anomaly region available.</p>
              )}
            </div>

            {/* AI Defect Analysis (Gemini Interpretation Layer) */}
            <VLMDefectAnalysisCard
              isPass={!isAnomalous}
              vlmAnalysis={inspection.vlm_analysis}
              onGenerate={handleGenerateVlm}
              isGenerating={isRetryingVlm}
            />
          </Card>

          {/* Expanded Feedback Section */}
          <Card className="p-6 space-y-4 bg-white border-industrial-200 shadow-sm">
            <div className="flex items-center justify-between border-b border-industrial-200 pb-3">
              <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-industrial-800 flex items-center space-x-1.5">
                <MessageSquare className="w-4 h-4 text-brand-600" />
                <span>Operator Feedback & Audit Trail</span>
              </h2>
              <span className="text-[10px] font-mono text-industrial-500 uppercase">Non-Destructive Logging</span>
            </div>

            {submitSuccess && (
              <div className="p-3 bg-pass-50 border border-pass-200 text-pass-700 text-xs rounded-lg font-medium flex items-center space-x-2">
                <Check className="w-4 h-4 text-pass-600 flex-shrink-0" />
                <span>Feedback submitted successfully! Preserved for future model learning.</span>
              </div>
            )}

            <form onSubmit={handleFeedbackSubmit} className="space-y-4 font-mono text-xs">
              {/* Section 1: PatchCore Detection Feedback */}
              <div className="space-y-2">
                <p className="font-bold text-industrial-900 text-[11px] uppercase tracking-wider">
                  1. PatchCore Anomaly Detection Feedback:
                </p>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setDetectionFeedback('correct')}
                    className={`py-2 px-3 text-xs font-bold rounded-lg border transition-all flex items-center justify-center space-x-1.5 ${
                      detectionFeedback === 'correct'
                        ? 'bg-pass-50 text-pass-700 border-pass-300 ring-2 ring-pass-500/20 shadow-sm'
                        : 'bg-industrial-50 text-industrial-600 border-industrial-200 hover:bg-industrial-100'
                    }`}
                  >
                    <span>✓ Correct Verdict ({isAnomalous ? 'REJECT' : 'PASS'})</span>
                  </button>

                  {isAnomalous ? (
                    <button
                      type="button"
                      onClick={() => setDetectionFeedback('false_positive')}
                      className={`py-2 px-3 text-xs font-bold rounded-lg border transition-all flex items-center justify-center space-x-1.5 ${
                        detectionFeedback === 'false_positive'
                          ? 'bg-reject-50 text-reject-700 border-reject-300 ring-2 ring-reject-500/20 shadow-sm'
                          : 'bg-industrial-50 text-industrial-600 border-industrial-200 hover:bg-industrial-100'
                      }`}
                    >
                      <span>False Positive (Actually GOOD)</span>
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => setDetectionFeedback('false_negative')}
                      className={`py-2 px-3 text-xs font-bold rounded-lg border transition-all flex items-center justify-center space-x-1.5 ${
                        detectionFeedback === 'false_negative'
                          ? 'bg-review-50 text-review-700 border-review-300 ring-2 ring-review-500/20 shadow-sm'
                          : 'bg-industrial-50 text-industrial-600 border-industrial-200 hover:bg-industrial-100'
                      }`}
                    >
                      <span>False Negative (Missed Defect)</span>
                    </button>
                  )}
                </div>
              </div>

              {/* Section 2: Gemini VLM Interpretation Feedback */}
              <div className="space-y-2 pt-2 border-t border-industrial-100">
                <p className="font-bold text-industrial-900 text-[11px] uppercase tracking-wider">
                  2. Gemini AI Defect Interpretation Feedback:
                </p>
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  {[
                    { id: 'correct' as const, label: '✓ Correct Analysis' },
                    { id: 'wrong_defect_type' as const, label: 'Wrong Defect Type' },
                    { id: 'wrong_location' as const, label: 'Wrong Location' },
                    { id: 'wrong_severity' as const, label: 'Wrong Severity' },
                  ].map((cat) => {
                    const isSelected = vlmFeedbackCategories.includes(cat.id);
                    return (
                      <button
                        key={cat.id}
                        type="button"
                        onClick={() => toggleVlmCategory(cat.id)}
                        className={`py-1.5 px-2.5 rounded-lg border text-left font-medium transition-all ${
                          isSelected
                            ? 'bg-brand-50 text-brand-700 border-brand-300 ring-1 ring-brand-500/30 font-bold'
                            : 'bg-industrial-50 text-industrial-600 border-industrial-200 hover:bg-industrial-100'
                        }`}
                      >
                        {cat.label}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Section 3: Conditional Correction Inputs */}
              {(vlmFeedbackCategories.includes('wrong_defect_type') || detectionFeedback === 'false_negative') && (
                <div className="space-y-1 pt-1">
                  <label className="text-[10px] font-bold text-industrial-700 uppercase">
                    Correct Defect Type:
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. Scratch, Stain, Hole, Chip, Crack..."
                    value={correctedDefectType}
                    onChange={(e) => setCorrectedDefectType(e.target.value)}
                    className="w-full px-3 py-2 text-xs border border-industrial-300 rounded-lg focus:ring-1 focus:ring-brand-500 focus:outline-none bg-white font-sans"
                  />
                </div>
              )}

              {vlmFeedbackCategories.includes('wrong_location') && (
                <div className="space-y-1">
                  <label className="text-[10px] font-bold text-industrial-700 uppercase">
                    Correct Defect Location:
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. Upper-right corner, central pad..."
                    value={correctedLocation}
                    onChange={(e) => setCorrectedLocation(e.target.value)}
                    className="w-full px-3 py-2 text-xs border border-industrial-300 rounded-lg focus:ring-1 focus:ring-brand-500 focus:outline-none bg-white font-sans"
                  />
                </div>
              )}

              {vlmFeedbackCategories.includes('wrong_severity') && (
                <div className="space-y-1">
                  <label className="text-[10px] font-bold text-industrial-700 uppercase">
                    Correct Defect Severity:
                  </label>
                  <select
                    value={correctedSeverity}
                    onChange={(e) => setCorrectedSeverity(e.target.value)}
                    className="w-full px-3 py-2 text-xs border border-industrial-300 rounded-lg focus:ring-1 focus:ring-brand-500 focus:outline-none bg-white font-sans"
                  >
                    <option value="">Select Severity Level...</option>
                    <option value="Low">Low</option>
                    <option value="Medium">Medium</option>
                    <option value="High">High</option>
                    <option value="Critical">Critical</option>
                  </select>
                </div>
              )}

              {/* Section 4: Comment Text Area */}
              <div className="space-y-1 pt-1">
                <label className="text-[10px] font-bold text-industrial-700 uppercase">
                  Additional Operator Notes (Optional):
                </label>
                <textarea
                  rows={2}
                  placeholder="Provide context for future model calibration or VLM prompt tuning..."
                  value={feedbackComment}
                  onChange={(e) => setFeedbackComment(e.target.value)}
                  className="w-full p-2.5 text-xs border border-industrial-300 rounded-lg focus:ring-1 focus:ring-brand-500 focus:outline-none bg-white font-sans"
                />
              </div>

              <Button
                type="submit"
                variant="primary"
                size="sm"
                className="w-full py-2"
                isLoading={isSubmittingFeedback}
                icon={<Send className="w-3.5 h-3.5" />}
              >
                Submit Operator Feedback
              </Button>
            </form>

            {/* Submitted feedback list / Audit Trail */}
            {feedbackList.length > 0 && (
              <div className="pt-4 border-t border-industrial-200 space-y-2">
                <span className="text-[11px] font-mono font-bold text-industrial-700 uppercase tracking-wider">
                  Saved Feedback Audit Trail ({feedbackList.length})
                </span>
                <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                  {feedbackList.map((f) => (
                    <div
                      key={f.id || f._id}
                      className="p-3 bg-industrial-50 rounded-lg border border-industrial-200 text-xs font-mono space-y-1.5"
                    >
                      <div className="flex justify-between items-center text-[10px] text-industrial-500 border-b border-industrial-200 pb-1">
                        <span className="font-bold text-industrial-800">
                          Detection: <span className="uppercase text-brand-700">{f.detection_feedback || f.feedback_type}</span>
                        </span>
                        <span>{f.created_at ? new Date(f.created_at).toLocaleString() : 'Just now'}</span>
                      </div>

                      {f.vlm_feedback_categories && f.vlm_feedback_categories.length > 0 && (
                        <div className="text-[10px] text-industrial-600 flex flex-wrap gap-1 items-center">
                          <span className="font-semibold text-industrial-500">VLM Feedback:</span>
                          {f.vlm_feedback_categories.map((c) => (
                            <span key={c} className="px-1.5 py-0.5 bg-brand-50 text-brand-700 border border-brand-200 rounded text-[9px] font-bold">
                              {c}
                            </span>
                          ))}
                        </div>
                      )}

                      {(f.corrected_defect_type || f.corrected_location || f.corrected_severity) && (
                        <div className="text-[11px] bg-white p-2 rounded border border-industrial-200 space-y-0.5 text-industrial-800">
                          {f.corrected_defect_type && <div><span className="text-industrial-400">Correct Type:</span> <strong>{f.corrected_defect_type}</strong></div>}
                          {f.corrected_location && <div><span className="text-industrial-400">Correct Location:</span> <strong>{f.corrected_location}</strong></div>}
                          {f.corrected_severity && <div><span className="text-industrial-400">Correct Severity:</span> <strong>{f.corrected_severity}</strong></div>}
                        </div>
                      )}

                      {f.comment && <p className="text-industrial-800 italic text-[11px]">"{f.comment}"</p>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>
      )}
    </div>
  );
};
