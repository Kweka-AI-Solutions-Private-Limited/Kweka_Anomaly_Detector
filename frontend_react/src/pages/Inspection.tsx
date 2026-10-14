import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import {
  Upload,
  ZoomIn,
  ZoomOut,
  Maximize2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Send,
  Cpu,
  Layers,
  Sliders,
  Check,
  Play,
  Power,
  RotateCcw,
  Info,
  X,
  FileImage,
  Sparkles,
  RefreshCw,
} from 'lucide-react';
import { getModels, activateModel, getModelVersions } from '../api/models';
import { runInspection, retryVlmAnalysis, retryInstanceVlmAnalysis } from '../api/inspections';
import { createInspectionRun, getInspectionRun } from '../api/runs';

import { submitFeedback } from '../api/feedback';
import { getStorageUrl } from '../api/client';
import { Model, ModelVersion, InspectionBatchItem, InspectionRun } from '../types';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { Button } from '../components/common/Button';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { VLMDefectAnalysisCard } from '../components/inspections/VLMDefectAnalysisCard';
import { MultiInstanceViewer } from '../components/inspections/MultiInstanceViewer';

export const Inspection: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  // Workflow Stage: 'setup' vs 'workspace'
  const [stage, setStage] = useState<'setup' | 'workspace'>('setup');

  // Inspection Mode State ('single_image' vs 'multi_instance')
  const [inspectionMode, setInspectionMode] = useState<'single_image' | 'multi_instance'>('multi_instance');

  // Models state
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [activeVersion, setActiveVersion] = useState<ModelVersion | null>(null);
  const [isLoadingModels, setIsLoadingModels] = useState<boolean>(true);
  const [isActivatingInline, setIsActivatingInline] = useState<boolean>(false);

  // Active Inspection Run state
  const [activeRun, setActiveRun] = useState<InspectionRun | null>(null);

  // Custom Threshold State (Inspection Level)
  const [thresholdMode, setThresholdMode] = useState<'default' | 'custom'>('default');
  const [customThresholdInput, setCustomThresholdInput] = useState<string>('');

  // Test files selected in setup screen
  const [testFileItems, setTestFileItems] = useState<{ id: string; file: File; preview: string }[]>([]);

  // Queue of inspection items processed during workspace
  const [batchItems, setBatchItems] = useState<InspectionBatchItem[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const [isProcessingBatch, setIsProcessingBatch] = useState<boolean>(false);
  const [processedCount, setProcessedCount] = useState<number>(0);

  // Viewer options
  const [activeTab, setActiveTab] = useState<'overlay' | 'original' | 'heatmap'>('overlay');
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(0.65);

  // Feedback state
  const [feedbackType, setFeedbackType] = useState<string>('correct');
  const [feedbackComment, setFeedbackComment] = useState<string>('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<boolean>(false);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState<boolean>(false);

  // VLM Retry State
  const [isRetryingVlm, setIsRetryingVlm] = useState<boolean>(false);

  // Error state
  const [error, setError] = useState<string | null>(null);

  // Scroll ref for Right Inspection Result panel to ensure reset on image change
  const resultPanelScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (resultPanelScrollRef.current) {
      resultPanelScrollRef.current.scrollTop = 0;
    }
  }, [selectedIndex]);

  // Fetch models on load and select active model
  const fetchModelsList = async () => {
    try {
      setIsLoadingModels(true);
      setError(null);
      const fetchedModels = await getModels();
      setModels(fetchedModels);

      const paramModelId = searchParams.get('model_id');
      const activeModels = fetchedModels.filter((m) => m.status === 'active');

      if (paramModelId && fetchedModels.some((m) => (m.id || m._id) === paramModelId)) {
        setSelectedModelId(paramModelId);
      } else if (activeModels.length > 0) {
        setSelectedModelId(activeModels[0].id || activeModels[0]._id!);
      } else if (fetchedModels.length > 0) {
        setSelectedModelId(fetchedModels[0].id || fetchedModels[0]._id!);
      }
    } catch (err: any) {
      setError('Failed to load inspection models.');
    } finally {
      setIsLoadingModels(false);
    }
  };

  useEffect(() => {
    fetchModelsList();
  }, [searchParams]);

  // Fetch active version metadata for selected model
  useEffect(() => {
    if (!selectedModelId) {
      setActiveVersion(null);
      return;
    }
    const currentModel = models.find((m) => (m.id || m._id) === selectedModelId);
    if (!currentModel) return;

    getModelVersions(selectedModelId)
      .then((versions) => {
        const found = versions.find((v) => (v.id || v._id) === currentModel.active_version_id) || versions[0] || null;
        setActiveVersion(found);
      })
      .catch(() => setActiveVersion(null));
  }, [selectedModelId, models]);

  const handleInlineActivate = async (modelId: string) => {
    try {
      setIsActivatingInline(true);
      setError(null);
      await activateModel(modelId);
      const fetchedModels = await getModels();
      setModels(fetchedModels);
    } catch (err: any) {
      const detail = err.response?.data?.detail || err.message || 'Model activation failed.';
      setError(`Activation error: ${detail}`);
    } finally {
      setIsActivatingInline(false);
    }
  };

  const selectedModel = models.find((m) => (m.id || m._id) === selectedModelId);

  // File Handlers for Test Image Upload in Setup
  const handleTestFilesAdded = (files: File[]) => {
    const newItems = files.map((file, idx) => ({
      id: `${Date.now()}-${idx}-${Math.random().toString(36).substr(2, 4)}`,
      file,
      preview: URL.createObjectURL(file),
    }));
    setTestFileItems((prev) => [...prev, ...newItems]);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleTestFilesAdded(Array.from(e.target.files));
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleTestFilesAdded(Array.from(e.dataTransfer.files));
    }
  };

  const removeTestFile = (index: number) => {
    setTestFileItems((prev) => {
      const updated = [...prev];
      URL.revokeObjectURL(updated[index].preview);
      updated.splice(index, 1);
      return updated;
    });
  };

  // Polling ref timer
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, [stopPolling]);

  // Run Batch Inspection Handler calling POST /api/inspection-runs
  const handleStartInspectionRun = async () => {
    if (!selectedModelId) {
      setError('Please select an active inspection model.');
      return;
    }
    if (testFileItems.length === 0) {
      setError('Please upload at least 1 TEST image to run anomaly detection.');
      return;
    }

    let thresholdOverride: number | undefined = undefined;
    if (thresholdMode === 'custom') {
      const parsed = parseFloat(customThresholdInput);
      if (isNaN(parsed) || parsed <= 0) {
        setError('Custom threshold must be a finite numeric value greater than 0.');
        return;
      }
      thresholdOverride = parsed;
    }

    setError(null);
    setStage('workspace');
    setIsProcessingBatch(true);
    setProcessedCount(0);

    // Initialize batch queue items
    const initialBatch: InspectionBatchItem[] = testFileItems.map((item) => ({
      id: item.id,
      file: item.file,
      previewUrl: item.preview,
      status: 'processing',
    }));
    setBatchItems(initialBatch);
    setSelectedIndex(0);

    try {
      const files = testFileItems.map((item) => item.file);
      const runResult = await createInspectionRun(selectedModelId, files, thresholdOverride, inspectionMode);
      setActiveRun(runResult);
      const runId = runResult.run_id || runResult.id || (runResult as any)._id;

      // Helper to update batch items from inspection records
      const updateBatchFromRun = (runData: InspectionRun, isTerminalRun: boolean = false) => {
        const inspections = runData.inspections || [];
        setBatchItems((prev) =>
          prev.map((item, idx) => {
            const insp: any =
              inspections.find((i: any) => i.filename === item.file.name) ||
              inspections[idx];
            if (insp) {
              const isTerminalInsp = insp.status === 'completed' || insp.status === 'review';
              return {
                ...item,
                status: insp.status === 'failed' ? 'error' : isTerminalInsp ? 'completed' : 'processing',
                result: insp as any,
                error: insp.status === 'failed' ? (insp.error || 'Inference failed') : undefined,
              };
            }
            if (isTerminalRun) {
              return {
                ...item,
                status: 'error',
                error: runData.error_count && runData.error_count > 0
                  ? 'Inference failed on the server.'
                  : 'Inference did not return a verdict',
              };
            }
            return item;
          })
        );
      };

      // Check if run already finished (e.g. synchronous execution mode)
      if (runResult.status === 'completed' || runResult.status === 'partial' || runResult.status === 'failed') {
        updateBatchFromRun(runResult, true);
        setProcessedCount(runResult.completed_images || testFileItems.length);
        setIsProcessingBatch(false);
        return;
      }

      // Live 1-second polling loop for background run progression
      stopPolling();
      pollingRef.current = setInterval(async () => {
        try {
          const runUpdate = await getInspectionRun(runId);
          setActiveRun(runUpdate);
          setProcessedCount(runUpdate.completed_images || 0);

          const isFinished = runUpdate.status === 'completed' || runUpdate.status === 'partial' || runUpdate.status === 'failed';
          updateBatchFromRun(runUpdate, isFinished);

          if (isFinished) {
            stopPolling();
            setIsProcessingBatch(false);
          }
        } catch (pollErr: any) {
          console.warn('Inspection run polling error:', pollErr);
        }
      }, 1000);

    } catch (err: any) {
      stopPolling();
      let detail = err.response?.data?.detail || err.message || 'Inspection run failed.';
      if (detail === 'Network Error') {
        detail = 'Network Error: Unable to connect to inspection service. The backend may be booting up or experiencing connectivity issues. Please retry in a few moments.';
      }
      setError(`Batch inspection run error: ${detail}`);
      setBatchItems((prev) =>
        prev.map((b) => ({ ...b, status: 'error', error: detail }))
      );
      setIsProcessingBatch(false);
    }
  };


  // Feedback Submission
  const currentItem = batchItems[selectedIndex];
  const handleFeedbackSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentItem?.result?.inspection_id) return;

    try {
      setIsSubmittingFeedback(true);
      await submitFeedback(currentItem.result.inspection_id, feedbackType, feedbackComment);
      setFeedbackSubmitted(true);
    } catch (err: any) {
      setError('Failed to submit feedback.');
    } finally {
      setIsSubmittingFeedback(false);
    }
  };

  // VLM Trigger / Retry Handler
  const handleGenerateVlm = async (force?: boolean) => {
    if (!currentItem?.result?.inspection_id) return;
    try {
      setIsRetryingVlm(true);
      const updated = await retryVlmAnalysis(currentItem.result.inspection_id, force);
      setBatchItems((prev) =>
        prev.map((item, idx) => {
          if (idx === selectedIndex) {
            return { ...item, result: updated };
          }
          return item;
        })
      );
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to generate Gemini AI analysis.');
    } finally {
      setIsRetryingVlm(false);
    }
  };

  // Retry single image inspection handler
  const handleRetrySingleItem = async (index: number) => {
    const item = batchItems[index];
    if (!item || !selectedModelId) return;

    setBatchItems((prev) =>
      prev.map((b, i) => (i === index ? { ...b, status: 'processing', error: undefined } : b))
    );

    try {
      let thresholdOverride: number | undefined = undefined;
      if (thresholdMode === 'custom') {
        const parsed = parseFloat(customThresholdInput);
        if (!isNaN(parsed) && parsed > 0) {
          thresholdOverride = parsed;
        }
      }
      const singleRun = await createInspectionRun(selectedModelId, [item.file], thresholdOverride, inspectionMode);
      const runId = singleRun.run_id || singleRun.id || (singleRun as any)._id;

      if (singleRun.status === 'completed' || singleRun.status === 'partial' || singleRun.status === 'failed') {
        const insp = (singleRun.inspections && singleRun.inspections[0]) as any;
        if (insp) {
          setBatchItems((prev) =>
            prev.map((b, i) =>
              i === index
                ? {
                    ...b,
                    status: insp.status === 'failed' ? 'error' : 'completed',
                    result: insp,
                    error: insp.status === 'failed' ? (insp.error || 'Inference failed') : undefined,
                  }
                : b
            )
          );
        } else {
          setBatchItems((prev) =>
            prev.map((b, i) =>
              i === index ? { ...b, status: 'error', error: 'Inference did not return a result' } : b
            )
          );
        }
        return;
      }

      // Polling single run
      const pollInterval = setInterval(async () => {
        try {
          const update = await getInspectionRun(runId);
          if (update.status === 'completed' || update.status === 'partial' || update.status === 'failed') {
            clearInterval(pollInterval);
            const insp = (update.inspections && update.inspections[0]) as any;
            setBatchItems((prev) =>
              prev.map((b, i) =>
                i === index
                  ? {
                      ...b,
                      status: insp && insp.status !== 'failed' ? 'completed' : 'error',
                      result: insp,
                      error: insp && insp.status === 'failed' ? (insp.error || 'Inference failed') : undefined,
                    }
                  : b
              )
            );
          }
        } catch (e) {
          clearInterval(pollInterval);
        }
      }, 1000);
    } catch (err: any) {
      const detail = err.response?.data?.detail || err.message || 'Retry failed';
      setBatchItems((prev) =>
        prev.map((b, i) => (i === index ? { ...b, status: 'error', error: detail } : b))
      );
    }
  };

  // Calculate Batch Summary Stats
  const passCount = batchItems.filter(
    (b) => b.result?.prediction?.status === 'normal' || b.result?.prediction?.status === 'PASS'
  ).length;
  const rejectCount = batchItems.filter(
    (b) => b.result?.prediction?.status === 'anomalous' || b.result?.prediction?.status === 'REJECT'
  ).length;

  if (isLoadingModels) {
    return <LoadingSpinner label="Loading inspection configuration..." size="lg" />;
  }

  return (
    <div className="space-y-4 w-full min-w-0">
      {/* STAGE 1: INSPECTION SETUP SCREEN */}
      {stage === 'setup' && (
        <div className="w-full max-w-6xl mx-auto space-y-6">
          {error && (
            <div className="p-3 bg-reject-50 border border-reject-200 text-reject-700 text-xs rounded-xl flex items-center justify-between">
              <span>{error}</span>
              <button onClick={() => setError(null)} className="text-reject-600 underline font-semibold">
                Dismiss
              </button>
            </div>
          )}
          {/* Header */}
          <div className="bg-white p-6 rounded-lg border border-industrial-200 shadow-sm">
            <h1 className="text-xl font-bold text-industrial-900 tracking-tight">New Inspection</h1>
            <p className="text-xs text-industrial-500 mt-1">
              Configure your inspection model and upload product test images before running PatchCore anomaly detection.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            {/* Left Setup Column (7 Cols) */}
            <div className="md:col-span-7 space-y-6">
              {/* 1. Model Selector Card */}
              <Card className="p-6 space-y-4">
                <div>
                  <label className="block text-xs font-mono font-bold text-industrial-700 uppercase tracking-wider mb-2">
                    1. Select Inspection Model
                  </label>
                  <select
                    value={selectedModelId}
                    onChange={(e) => setSelectedModelId(e.target.value)}
                    className="w-full px-4 py-3 bg-white text-industrial-900 text-base font-extrabold border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none shadow-sm cursor-pointer"
                  >
                    {models.map((m) => (
                      <option key={m.id || m._id} value={m.id || m._id}>
                        {m.name} — [{(m.status || 'draft').toUpperCase()}]
                      </option>
                    ))}
                  </select>
                </div>

                {selectedModel && (
                  <div className="space-y-3">
                    {/* Non-ACTIVE Usability Warning Banner */}
                    {selectedModel.status !== 'active' && (
                      <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl space-y-2.5 shadow-sm">
                        <div className="flex items-center space-x-2 font-extrabold text-amber-900 text-sm">
                          <AlertTriangle className="w-4.5 h-4.5 text-amber-600 flex-shrink-0" />
                          <span>Model Not Usable ({selectedModel.status.toUpperCase()})</span>
                        </div>
                        <p className="text-industrial-700 text-xs leading-relaxed font-medium">
                          Model <strong className="text-industrial-900 font-extrabold">{selectedModel.name}</strong> is currently in{' '}
                          <strong className="text-amber-800 font-extrabold">{selectedModel.status.toUpperCase()}</strong> state and cannot execute inspections.
                        </p>
                        {selectedModel.status === 'inactive' && (
                          <Button
                            variant="primary"
                            size="sm"
                            className="w-full text-xs font-bold shadow-sm"
                            isLoading={isActivatingInline}
                            onClick={() => handleInlineActivate(selectedModelId)}
                            icon={<Power className="w-3.5 h-3.5 text-white" />}
                          >
                            Activate Model Now
                          </Button>
                        )}
                        {selectedModel.status === 'draft' && (
                          <Button
                            variant="outline"
                            size="sm"
                            className="w-full text-xs font-bold shadow-sm"
                            onClick={() => navigate(`/models/${selectedModelId}/build`)}
                          >
                            Configure & Build Model
                          </Button>
                        )}
                      </div>
                    )}

                    <div className="bg-industrial-50 p-4 rounded-xl border border-industrial-200 grid grid-cols-2 gap-3.5 text-xs font-mono">
                      <div>
                        <span className="text-industrial-600 block text-[11px] uppercase tracking-wider font-extrabold mb-1">STATUS</span>
                        <Badge status={selectedModel.status} size="sm" />
                      </div>
                      <div>
                        <span className="text-industrial-600 block text-[11px] uppercase tracking-wider font-extrabold mb-1">REFERENCE BANK</span>
                        <span className="font-extrabold text-industrial-900 text-sm block">
                          {selectedModel.reference_image_count} normal images
                        </span>
                      </div>
                      <div>
                        <span className="text-industrial-600 block text-[11px] uppercase tracking-wider font-extrabold mb-1">ALGORITHM</span>
                        <span className="font-extrabold text-brand-700 text-sm block">PatchCore WRN-50</span>
                      </div>
                      <div>
                        <span className="text-industrial-600 block text-[11px] uppercase tracking-wider font-extrabold mb-1">ACTIVE VERSION</span>
                        <span className="font-extrabold text-industrial-900 text-sm block font-mono">
                          {activeVersion
                            ? `v${activeVersion.version_number} (${activeVersion.status.toUpperCase()})`
                            : selectedModel.active_version_id
                              ? 'Active'
                              : 'None'}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </Card>

              {/* 1.5 Inspection Mode Selection Card */}
              <Card className="p-6 space-y-4">
                <div>
                  <h3 className="text-xs font-mono font-bold text-industrial-700 uppercase tracking-wider">
                    Inspection Mode
                  </h3>
                  <p className="text-xs text-industrial-500 mt-1">
                    Select single-product inspection or multi-product instance inspection.
                  </p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => setInspectionMode('single_image')}
                    className={`p-4 rounded-xl border text-left transition-all ${inspectionMode === 'single_image'
                        ? 'border-brand-500 bg-brand-50/50 ring-2 ring-brand-500/20'
                        : 'border-industrial-200 bg-white hover:border-industrial-300'
                      }`}
                  >
                    <div className="flex items-center space-x-2 font-extrabold text-industrial-900 text-sm">
                      <FileImage className="w-4 h-4 text-brand-600" />
                      <span>Single Product</span>
                    </div>
                    <p className="text-[11px] text-industrial-500 mt-1.5 leading-snug">
                      1 item per uploaded image. Standard PatchCore anomaly detection.
                    </p>
                  </button>

                  <button
                    type="button"
                    onClick={() => setInspectionMode('multi_instance')}
                    className={`p-4 rounded-xl border text-left transition-all ${inspectionMode === 'multi_instance'
                        ? 'border-brand-500 bg-brand-50/50 ring-2 ring-brand-500/20'
                        : 'border-industrial-200 bg-white hover:border-industrial-300'
                      }`}
                  >
                    <div className="flex items-center space-x-2 font-extrabold text-industrial-900 text-sm">
                      <Layers className="w-4 h-4 text-brand-600" />
                      <span>Multi-Product (Pure-Gemini Engine)</span>
                    </div>
                    <p className="text-[11px] text-industrial-500 mt-1.5 leading-snug">
                      Inspect multiple physical products in one image directly via Gemini VLM. No reference images or PatchCore required.
                    </p>
                  </button>
                </div>
              </Card>

              {/* 2. TEST Image Upload Box */}
              <Card className="p-6 space-y-4">
                <div>
                  <h3 className="text-xs font-mono font-bold text-industrial-700 uppercase tracking-wider">
                    2. Test Images Upload
                  </h3>
                  <p className="text-xs text-industrial-500 mt-1">
                    Upload the products/images you want to inspect (TEST images).
                  </p>
                </div>

                <div
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={handleDrop}
                  className="border-2 border-dashed border-industrial-300 hover:border-brand-500 rounded-lg p-6 text-center bg-industrial-50 hover:bg-white transition-all cursor-pointer group"
                >
                  <input
                    type="file"
                    multiple
                    accept="image/png,image/jpeg,image/jpg"
                    onChange={handleFileInputChange}
                    className="hidden"
                    id="test-upload-input"
                  />
                  <label htmlFor="test-upload-input" className="cursor-pointer block space-y-2">
                    <div className="w-10 h-10 bg-white rounded-full border border-industrial-200 flex items-center justify-center mx-auto text-industrial-500 group-hover:text-brand-600 shadow-sm">
                      <Upload className="w-5 h-5" />
                    </div>
                    <p className="text-xs font-semibold text-industrial-900">
                      Drag & drop test images or click to browse
                    </p>
                    <p className="text-[10px] text-industrial-400 font-mono">PNG • JPG • JPEG</p>
                  </label>
                </div>

                {/* Thumbnails grid */}
                {testFileItems.length > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs font-mono text-industrial-600">
                      <span>Uploaded Test Images ({testFileItems.length})</span>
                      <button
                        type="button"
                        onClick={() => setTestFileItems([])}
                        className="text-reject-600 hover:underline"
                      >
                        Clear
                      </button>
                    </div>

                    <div className="grid grid-cols-4 sm:grid-cols-5 gap-2 max-h-48 overflow-y-auto p-1 border border-industrial-200 rounded-md bg-white">
                      {testFileItems.map((item, idx) => (
                        <div
                          key={item.id}
                          className="relative group rounded border border-industrial-200 overflow-hidden bg-industrial-50 h-16"
                        >
                          <img src={item.preview} alt="Test" className="w-full h-full object-cover" />
                          <button
                            type="button"
                            onClick={() => removeTestFile(idx)}
                            className="absolute top-0.5 right-0.5 bg-industrial-900/80 text-white p-0.5 rounded-full opacity-0 group-hover:opacity-100 transition-opacity hover:bg-reject-600"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </Card>

              {/* 3. Anomaly Threshold Selection Card */}
              <Card className="p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-mono font-bold text-industrial-700 uppercase tracking-wider">
                    3. Threshold Selection
                  </h3>
                  <span className="text-[11px] font-mono text-industrial-500 font-medium">
                    {thresholdMode === 'custom' ? 'Custom Override' : 'Model Default'}
                  </span>
                </div>

                <div className="space-y-3 font-mono text-xs">
                  <label className={`flex items-center justify-between p-3.5 rounded-xl border cursor-pointer transition-all ${thresholdMode === 'default'
                      ? 'border-brand-500 bg-brand-50/40 ring-1 ring-brand-500'
                      : 'border-industrial-200 bg-industrial-50 hover:bg-industrial-100/60'
                    }`}>
                    <div className="flex items-center space-x-3">
                      <input
                        type="radio"
                        name="thresholdMode"
                        value="default"
                        checked={thresholdMode === 'default'}
                        onChange={() => setThresholdMode('default')}
                        className="w-4 h-4 text-brand-600 focus:ring-brand-500"
                      />
                      <span className="font-extrabold text-industrial-900">Model Default</span>
                    </div>
                    <span className="font-bold text-brand-700 bg-white px-2.5 py-1 rounded-md border border-brand-200 text-xs shadow-xs">
                      {activeVersion?.calibration?.threshold !== undefined
                        ? activeVersion.calibration.threshold.toFixed(2)
                        : '24.54'}
                    </span>
                  </label>

                  <label className={`flex flex-col p-3.5 rounded-xl border cursor-pointer transition-all space-y-2.5 ${thresholdMode === 'custom'
                      ? 'border-brand-500 bg-brand-50/40 ring-1 ring-brand-500'
                      : 'border-industrial-200 bg-industrial-50 hover:bg-industrial-100/60'
                    }`}>
                    <div className="flex items-center space-x-3">
                      <input
                        type="radio"
                        name="thresholdMode"
                        value="custom"
                        checked={thresholdMode === 'custom'}
                        onChange={() => setThresholdMode('custom')}
                        className="w-4 h-4 text-brand-600 focus:ring-brand-500"
                      />
                      <span className="font-extrabold text-industrial-900">Custom Threshold</span>
                    </div>

                    {thresholdMode === 'custom' && (
                      <div className="pt-1 pl-7 flex flex-col sm:flex-row sm:items-center gap-3">
                        <input
                          type="number"
                          step="0.01"
                          min="0.01"
                          value={customThresholdInput}
                          onChange={(e) => {
                            setCustomThresholdInput(e.target.value);
                            setError(null);
                          }}
                          placeholder="e.g. 20.00"
                          className="w-36 px-3 py-1.5 bg-white border border-industrial-300 rounded-lg text-industrial-900 font-bold focus:ring-2 focus:ring-brand-500 outline-none text-sm shadow-xs"
                        />
                        <span className="text-[11px] text-industrial-500 font-sans leading-tight">
                          Applied to this inspection run only (no new model version created).
                        </span>
                      </div>
                    )}
                  </label>

                  {/* Sensitivity Guidance Note */}
                  <div className="p-3 bg-brand-50/60 border border-brand-200 rounded-lg text-[11px] text-industrial-700 space-y-1">
                    <p className="font-bold text-brand-900">Threshold Sensitivity Behavior:</p>
                    <p className="leading-relaxed">
                      • <strong>Higher threshold (e.g. 50.0)</strong> = Less sensitive (only stronger anomalies flagged as REJECT).<br />
                      • <strong>Lower threshold (e.g. 15.0)</strong> = More sensitive (weaker anomalies flagged as REJECT).<br />
                      • <strong>Scope:</strong> Applies strictly to <em>Single-Product PatchCore (Pipeline A)</em>. Multi-Product Pure-Gemini (Pipeline B) decisions are evaluated directly by Gemini VLM.
                    </p>
                  </div>
                </div>
              </Card>
            </div>

            {/* Right Summary & Run Column (5 Cols) */}
            <div className="md:col-span-5 space-y-6">
              <Card className="p-6 space-y-5">
                <h3 className="text-xs font-mono font-bold text-industrial-700 uppercase tracking-wider">
                  Inspection Summary
                </h3>

                <div className="bg-industrial-50 p-4 rounded-lg border border-industrial-200 space-y-3 font-mono text-xs">
                  <div className="flex justify-between items-center">
                    <span className="text-industrial-500">Model:</span>
                    <span className="font-bold text-industrial-900 truncate max-w-[160px]">
                      {selectedModel?.name || 'None'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-industrial-500">Version:</span>
                    <span className="font-semibold text-industrial-800 font-mono">
                      {activeVersion ? `v${activeVersion.version_number}` : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-industrial-500">Test Images:</span>
                    <span className="font-bold text-brand-600 text-sm">{testFileItems.length}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-industrial-500">Threshold:</span>
                    <span className="font-extrabold text-industrial-900 font-mono">
                      {thresholdMode === 'custom'
                        ? `${parseFloat(customThresholdInput || '0').toFixed(2)} (Custom)`
                        : `${activeVersion?.calibration?.threshold !== undefined ? activeVersion.calibration.threshold.toFixed(2) : '—'} (Model Default)`}
                    </span>
                  </div>
                </div>

                <Button
                  variant="primary"
                  size="lg"
                  className="w-full text-sm font-bold shadow-lg shadow-brand-600/30"
                  disabled={testFileItems.length === 0 || !selectedModelId || selectedModel?.status !== 'active'}
                  onClick={handleStartInspectionRun}
                  icon={<Play className="w-4 h-4 fill-white" />}
                >
                  {selectedModel?.status !== 'active'
                    ? `Model ${selectedModel?.status ? selectedModel.status.toUpperCase() : 'Not Active'}`
                    : `Run Inspection (${testFileItems.length} Images)`}
                </Button>
              </Card>
            </div>
          </div>
        </div>
      )}

      {/* STAGE 2: PROCESSING & INSPECTION WORKSPACE SCREEN */}
      {stage === 'workspace' && (
        <div className="space-y-4 min-h-[calc(100vh-120px)] flex flex-col">
          {/* Top Bar with Setup Reset & Batch Progress */}
          <div className="bg-white p-3 sm:p-4 rounded-xl border border-industrial-200 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-3 shrink-0">
            <div className="flex flex-wrap items-center gap-2.5 sm:gap-3">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setStage('setup')}
                icon={<ChevronLeft className="w-4 h-4" />}
              >
                Setup New Inspection
              </Button>
              <div className="hidden sm:block h-4 w-px bg-industrial-200" />
              <div className="min-w-0">
                <span className="text-[10px] font-mono text-industrial-400 uppercase font-semibold block">
                  Active Model
                </span>
                <span className="text-xs font-bold text-industrial-900 truncate block max-w-[200px]">
                  {selectedModel?.name}
                </span>
              </div>
            </div>

            {/* Batch Progress Status */}
            {isProcessingBatch ? (
              <div className="flex items-center space-x-2.5 bg-brand-50 border border-brand-200 px-3 py-1.5 rounded-lg text-xs font-mono">
                <div className="w-3.5 h-3.5 border-2 border-brand-600 border-t-transparent rounded-full animate-spin shrink-0" />
                <span className="text-brand-800 font-semibold truncate">
                  Analyzing test images... {processedCount} / {batchItems.length} completed
                </span>
              </div>
            ) : (
              <div className="flex flex-wrap items-center gap-1.5 sm:gap-2 text-xs font-mono">
                <span className="font-bold text-industrial-700">
                  {batchItems.length} Images Inspected:
                </span>
                <span className="bg-pass-50 text-pass-700 px-2 py-0.5 rounded font-bold border border-pass-200">
                  {passCount} PASS
                </span>
                <span className="bg-reject-50 text-reject-700 px-2 py-0.5 rounded font-bold border border-reject-200">
                  {rejectCount} REJECT
                </span>
                {batchItems.filter(b => b.status === 'error').length > 0 && (
                  <span className="bg-reject-100 text-reject-800 px-2 py-0.5 rounded font-bold border border-reject-300">
                    {batchItems.filter(b => b.status === 'error').length} FAILED
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Optional inline workspace error notice */}
          {error && (
            <div className="p-3 bg-reject-50 border border-reject-200 text-reject-700 text-xs rounded-xl flex items-center justify-between shrink-0">
              <span className="truncate mr-2">{error}</span>
              <button onClick={() => setError(null)} className="text-reject-600 underline font-semibold shrink-0">
                Dismiss
              </button>
            </div>
          )}

          {/* Main 3-Column Layout */}
          <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-3 sm:gap-4 min-h-0">
            {/* LEFT PANEL: Test Queue (2 Cols on desktop, horizontal carousel reel on mobile/tablet) */}
            <div className="lg:col-span-2 bg-white rounded-xl border border-industrial-200 flex flex-col shadow-sm overflow-hidden shrink-0">
              <div className="p-2.5 sm:p-3 border-b border-industrial-200 bg-industrial-50 flex items-center justify-between shrink-0">
                <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-industrial-700 truncate">
                  Queue ({batchItems.length})
                </h3>
                <span className="text-[10px] text-industrial-400 font-mono lg:hidden">
                  Swipe horizontal →
                </span>
              </div>

              <div className="p-2 flex lg:flex-col overflow-x-auto lg:overflow-y-auto gap-2 scrollbar-thin">
                {batchItems.map((item, idx) => {
                  const isSelected = idx === selectedIndex;
                  const predStatus = item.result?.prediction?.status;

                  return (
                    <div
                      key={item.id}
                      onClick={() => {
                        setSelectedIndex(idx);
                        setFeedbackSubmitted(false);
                      }}
                      className={`shrink-0 w-44 sm:w-52 lg:w-full p-2 sm:p-2.5 rounded-lg border flex items-center space-x-2.5 cursor-pointer transition-all ${
                        isSelected
                          ? 'border-brand-500 bg-brand-50/60 shadow-sm ring-1 ring-brand-500'
                          : item.status === 'error'
                            ? 'border-reject-300 bg-reject-50/40 hover:border-reject-400'
                            : 'border-industrial-200 hover:border-industrial-300 bg-white'
                      }`}
                    >
                      <div className="w-10 h-10 rounded border border-industrial-200 overflow-hidden bg-industrial-100 shrink-0 relative">
                        <img src={item.previewUrl} alt="Thumbnail" className="w-full h-full object-cover" />
                        {item.status === 'error' && (
                          <div className="absolute inset-0 bg-reject-900/60 flex items-center justify-center">
                            <AlertTriangle className="w-3.5 h-3.5 text-white" />
                          </div>
                        )}
                      </div>

                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-mono font-semibold text-industrial-900 truncate">
                          {item.file.name}
                        </p>
                        <div className="mt-1 flex items-center space-x-1.5">
                          {item.status === 'processing' && (
                            <span className="text-[10px] font-mono text-brand-600 animate-pulse font-semibold">
                              Analyzing...
                            </span>
                          )}
                          {item.status === 'completed' && (
                            <>
                              <Badge status={predStatus || (item.result?.overall_prediction?.status) || 'normal'} size="sm" />
                              <span className="text-[11px] font-mono text-industrial-600">
                                {item.result?.prediction?.anomaly_score !== undefined
                                  ? item.result.prediction.anomaly_score.toFixed(1)
                                  : (item.result?.overall_prediction?.max_anomaly_score?.toFixed(1) ?? '')}
                              </span>
                            </>
                          )}
                          {item.status === 'error' && (
                            <span className="text-[10px] font-mono text-reject-600 font-bold uppercase">
                              Error
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* CENTER + RIGHT PANELS: Multi-Instance vs Single-Image Viewer */}
            {currentItem && (
              currentItem.result?.inspection_mode === 'multi_instance' ||
              (currentItem.result as any)?.result?.inspection_mode === 'multi_instance' ||
              (currentItem as any)?.inspection_mode === 'multi_instance' ||
              inspectionMode === 'multi_instance' ||
              (currentItem.result?.instances && currentItem.result.instances.length > 0) ||
              ((currentItem.result as any)?.result?.instances && (currentItem.result as any).result.instances.length > 0)
            ) ? (
              <div className="lg:col-span-10 bg-white rounded-lg border border-industrial-200 p-4 overflow-y-auto shadow-sm">
                <MultiInstanceViewer
                  result={currentItem.result || (currentItem.result as any)?.result}
                  onTriggerInstanceVlm={async (instanceId) => {
                    const inspId = currentItem.result?.inspection_id || (currentItem.result as any)?.id;
                    if (inspId) {
                      const updated = await retryInstanceVlmAnalysis(inspId, instanceId);
                      setBatchItems((prev) =>
                        prev.map((b, i) => (i === selectedIndex ? { ...b, result: updated } : b))
                      );
                    }
                  }}
                />
              </div>
            ) : (
              <>
                {/* CENTER PANEL: Image Viewer (6 Cols on desktop) */}
                <div className="lg:col-span-6 bg-white rounded-xl border border-industrial-200 flex flex-col shadow-sm overflow-hidden min-h-[360px] sm:min-h-[460px] lg:min-h-0">
                  {/* Toolbar */}
                  <div className="p-2 sm:p-2.5 border-b border-industrial-200 bg-industrial-50 flex flex-wrap items-center justify-between gap-2 shrink-0">
                    <div className="flex items-center space-x-1 bg-white rounded-lg border border-industrial-200 p-0.5">
                      <button
                        onClick={() => setActiveTab('overlay')}
                        className={`px-2.5 sm:px-3 py-1 text-[11px] sm:text-xs font-semibold rounded-md transition-colors ${
                          activeTab === 'overlay' ? 'bg-brand-600 text-white shadow-xs' : 'text-industrial-600 hover:text-industrial-900'
                        }`}
                      >
                        Heatmap Overlay
                      </button>
                      <button
                        onClick={() => setActiveTab('original')}
                        className={`px-2.5 sm:px-3 py-1 text-[11px] sm:text-xs font-semibold rounded-md transition-colors ${
                          activeTab === 'original' ? 'bg-brand-600 text-white shadow-xs' : 'text-industrial-600 hover:text-industrial-900'
                        }`}
                      >
                        Original Image
                      </button>
                      <button
                        onClick={() => setActiveTab('heatmap')}
                        className={`px-2.5 sm:px-3 py-1 text-[11px] sm:text-xs font-semibold rounded-md transition-colors ${
                          activeTab === 'heatmap' ? 'bg-brand-600 text-white shadow-xs' : 'text-industrial-600 hover:text-industrial-900'
                        }`}
                      >
                        Heatmap Only
                      </button>
                    </div>

                    <div className="flex items-center space-x-2 sm:space-x-3">
                      {activeTab === 'overlay' && (
                        <div className="flex items-center space-x-1.5 text-xs font-mono text-industrial-600">
                          <Sliders className="w-3.5 h-3.5 hidden sm:inline" />
                          <span className="text-[11px]">Opacity:</span>
                          <input
                            type="range"
                            min="0.1"
                            max="1.0"
                            step="0.05"
                            value={heatmapOpacity}
                            onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                            className="w-16 sm:w-20 h-1 bg-industrial-200 rounded-lg appearance-none cursor-pointer accent-brand-600"
                          />
                        </div>
                      )}

                      <div className="flex items-center space-x-1">
                        <button
                          onClick={() => setZoomLevel((z) => Math.max(0.5, z - 0.25))}
                          className="p-1 text-industrial-500 hover:text-industrial-900 hover:bg-industrial-100 rounded"
                          title="Zoom Out"
                        >
                          <ZoomOut className="w-4 h-4" />
                        </button>
                        <span className="text-xs font-mono text-industrial-600 w-9 text-center">
                          {Math.round(zoomLevel * 100)}%
                        </span>
                        <button
                          onClick={() => setZoomLevel((z) => Math.min(3, z + 0.25))}
                          className="p-1 text-industrial-500 hover:text-industrial-900 hover:bg-industrial-100 rounded"
                          title="Zoom In"
                        >
                          <ZoomIn className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setZoomLevel(1)}
                          className="p-1 text-industrial-500 hover:text-industrial-900 hover:bg-industrial-100 rounded"
                          title="Reset Zoom"
                        >
                          <Maximize2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Viewport Area */}
                  <div className="flex-1 bg-industrial-900 relative overflow-hidden flex items-center justify-center p-3 sm:p-4 min-h-[300px] sm:min-h-[420px] lg:min-h-0">
                    {/* Viewport Floating Error Banner */}
                    {currentItem?.status === 'error' && (
                      <div className="absolute top-3 left-3 right-3 z-20 bg-reject-950/90 border border-reject-500/50 backdrop-blur-md rounded-xl p-3 flex items-center justify-between shadow-2xl animate-in fade-in slide-in-from-top-1">
                        <div className="flex items-center space-x-2.5 min-w-0">
                          <div className="p-1.5 rounded-lg bg-reject-500/20 text-reject-400 shrink-0">
                            <AlertTriangle className="w-4 h-4" />
                          </div>
                          <div className="min-w-0">
                            <span className="text-xs font-bold text-reject-200 font-mono uppercase tracking-wide block">
                              Inference Error
                            </span>
                            <p className="text-[11px] text-reject-300 font-mono truncate">
                              {currentItem.error || (currentItem.result as any)?.error || 'Anomaly detection failed for this image.'}
                            </p>
                          </div>
                        </div>
                        <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-reject-900/90 border border-reject-700 text-reject-300 uppercase shrink-0">
                          Failed
                        </span>
                      </div>
                    )}

                    {/* Viewport Floating Verdict Overlay for Completed Inspections */}
                    {currentItem?.status === 'completed' && currentItem?.result && (
                      <div className="absolute top-3 left-3 z-20 pointer-events-none">
                        {currentItem.result.prediction?.status === 'anomalous' ||
                        currentItem.result.prediction?.status === 'REJECT' ||
                        (currentItem.result.prediction?.anomaly_score !== undefined &&
                          currentItem.result.prediction?.threshold !== undefined &&
                          currentItem.result.prediction.anomaly_score >= currentItem.result.prediction.threshold) ||
                        currentItem.result.overall_prediction?.status === 'anomalous' ||
                        currentItem.result.overall_prediction?.status === 'REJECT' ? (
                          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-reject-950/90 border border-reject-500/60 backdrop-blur-md shadow-lg text-white font-mono">
                            <XCircle className="w-4 h-4 text-reject-400 shrink-0" />
                            <span className="text-xs font-black tracking-wider text-reject-300 uppercase">REJECT</span>
                            <span className="text-industrial-500 text-[10px]">|</span>
                            <span className="text-[11px] font-bold text-reject-200">
                              Score: {(currentItem.result.prediction?.anomaly_score ?? currentItem.result.overall_prediction?.max_anomaly_score ?? 0).toFixed(1)}
                            </span>
                            <span className="text-industrial-500 text-[10px]">/</span>
                            <span className="text-[10px] text-industrial-400">
                              Thr: {(currentItem.result.prediction?.threshold ?? (selectedModel as any)?.active_version?.calibration?.threshold ?? 21.43).toFixed(1)}
                            </span>
                          </div>
                        ) : (
                          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-pass-950/90 border border-pass-500/60 backdrop-blur-md shadow-lg text-white font-mono">
                            <CheckCircle2 className="w-4 h-4 text-pass-400 shrink-0" />
                            <span className="text-xs font-black tracking-wider text-pass-300 uppercase">PASS</span>
                            <span className="text-industrial-500 text-[10px]">|</span>
                            <span className="text-[11px] font-bold text-pass-200">
                              Score: {(currentItem.result.prediction?.anomaly_score ?? currentItem.result.overall_prediction?.max_anomaly_score ?? 0).toFixed(1)}
                            </span>
                            <span className="text-industrial-500 text-[10px]">/</span>
                            <span className="text-[10px] text-industrial-400">
                              Thr: {(currentItem.result.prediction?.threshold ?? (selectedModel as any)?.active_version?.calibration?.threshold ?? 21.43).toFixed(1)}
                            </span>
                          </div>
                        )}
                      </div>
                    )}

                    {!currentItem ? (
                      <div className="text-center text-industrial-500 font-mono text-xs">
                        Select a test image from the queue.
                      </div>
                    ) : currentItem.status === 'processing' ? (
                      <div className="text-center space-y-3">
                        <div className="w-10 h-10 border-4 border-brand-500 border-t-transparent rounded-full animate-spin mx-auto" />
                        <p className="text-xs font-mono text-brand-400">Running PatchCore Feature Comparison...</p>
                      </div>
                    ) : (
                      <div
                        className="relative transition-transform duration-200 ease-out max-h-full max-w-full"
                        style={{ transform: `scale(${zoomLevel})` }}
                      >
                        <img
                          src={currentItem.previewUrl}
                          alt="Original"
                          className={`max-h-[320px] sm:max-h-[440px] w-auto max-w-full object-contain rounded-lg shadow-lg ${
                            activeTab === 'heatmap' ? 'opacity-0' : 'opacity-100'
                          }`}
                        />

                        {currentItem.result?.localization?.heatmap_uri && activeTab !== 'original' && (
                          <img
                            src={getStorageUrl(currentItem.result.localization.heatmap_uri)}
                            alt="Heatmap"
                            className="absolute inset-0 w-full h-full object-contain rounded-lg pointer-events-none"
                            style={{
                              opacity: activeTab === 'heatmap' ? 1.0 : heatmapOpacity,
                            }}
                          />
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* RIGHT PANEL: Result & Metadata (4 Cols on desktop) */}
                <div className="lg:col-span-4 bg-white rounded-xl border border-industrial-200 flex flex-col shadow-sm overflow-hidden">
                  {/* Pinned Card Header */}
                  <div className="p-3 border-b border-industrial-200 bg-industrial-50 shrink-0 flex items-center justify-between">
                    <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-industrial-700">
                      Inspection Result
                    </h3>
                    {currentItem?.status === 'completed' && currentItem.result && (
                      <div className="flex items-center space-x-2">
                        <span className="text-[10px] font-mono text-industrial-500 truncate max-w-[110px]">
                          {currentItem.file.name}
                        </span>
                        <Badge
                          status={
                            currentItem.result.prediction?.status === 'anomalous' ||
                            currentItem.result.prediction?.status === 'REJECT' ||
                            (currentItem.result.prediction?.anomaly_score !== undefined &&
                              currentItem.result.prediction?.threshold !== undefined &&
                              currentItem.result.prediction.anomaly_score >= currentItem.result.prediction.threshold) ||
                            currentItem.result.overall_prediction?.status === 'anomalous' ||
                            currentItem.result.overall_prediction?.status === 'REJECT'
                              ? 'anomalous'
                              : 'normal'
                          }
                          size="sm"
                        />
                      </div>
                    )}
                  </div>

                  {/* PINNED VERDICT & KEY METRICS (Always visible, never scrolled away) */}
                  {currentItem?.status === 'completed' && currentItem.result && (
                    <div className="p-3.5 border-b border-industrial-200 bg-industrial-50/50 shrink-0 space-y-2.5">
                      {/* Status Banner */}
                      {currentItem.result.prediction?.status === 'anomalous' ||
                      currentItem.result.prediction?.status === 'REJECT' ||
                      (currentItem.result.prediction?.anomaly_score !== undefined &&
                        currentItem.result.prediction?.threshold !== undefined &&
                        currentItem.result.prediction.anomaly_score >= currentItem.result.prediction.threshold) ||
                      currentItem.result.overall_prediction?.status === 'anomalous' ||
                      currentItem.result.overall_prediction?.status === 'REJECT' ? (
                        <div className="p-3 bg-reject-50 border border-reject-200 rounded-lg text-center space-y-1 shadow-2xs">
                          <div className="flex items-center justify-center space-x-1.5">
                            <XCircle className="w-5 h-5 text-reject-600" />
                            <h2 className="text-base font-extrabold text-reject-800 tracking-tight font-mono uppercase">
                              REJECT
                            </h2>
                          </div>
                          <p className="text-[11px] font-bold text-reject-600 font-mono tracking-wide">
                            ANOMALY DETECTED
                          </p>
                        </div>
                      ) : (
                        <div className="p-3 bg-pass-50 border border-pass-200 rounded-lg text-center space-y-1 shadow-2xs">
                          <div className="flex items-center justify-center space-x-1.5">
                            <CheckCircle2 className="w-5 h-5 text-pass-600" />
                            <h2 className="text-base font-extrabold text-pass-800 tracking-tight font-mono uppercase">
                              PASS
                            </h2>
                          </div>
                          <p className="text-[11px] font-bold text-pass-600 font-mono tracking-wide">
                            PRODUCT NORMAL
                          </p>
                        </div>
                      )}

                      {/* Scores & Threshold Grid */}
                      <div className="grid grid-cols-2 gap-2 font-mono">
                        <div className="p-2.5 bg-white rounded-lg border border-industrial-200 shadow-2xs">
                          <span className="text-[10px] text-industrial-500 uppercase font-semibold">Anomaly Score</span>
                          <p className="text-sm sm:text-base font-extrabold text-industrial-900 mt-0.5">
                            {(currentItem.result.prediction?.anomaly_score ?? currentItem.result.overall_prediction?.max_anomaly_score)?.toFixed(2) ?? 'N/A'}
                          </p>
                        </div>
                        <div className="p-2.5 bg-white rounded-lg border border-industrial-200 shadow-2xs">
                          <span className="text-[10px] text-industrial-500 uppercase font-semibold">Threshold</span>
                          <p className="text-sm sm:text-base font-extrabold text-industrial-700 mt-0.5">
                            {(currentItem.result.prediction?.threshold ?? (selectedModel as any)?.active_version?.calibration?.threshold)?.toFixed(2) ?? '21.43'}
                          </p>
                        </div>
                      </div>

                      {/* Compact Severity & Latency Bar */}
                      <div className="flex items-center justify-between px-2.5 py-1.5 bg-white rounded-lg border border-industrial-200 text-[11px] font-mono">
                        <span className="text-industrial-500">
                          Severity: <strong className="text-industrial-800 capitalize font-bold">{currentItem.result.prediction?.severity || 'none'}</strong>
                        </span>
                        <span className="text-industrial-500">
                          Latency: <strong className="text-industrial-800 font-bold">{currentItem.result.processing_time_ms || 32} ms</strong>
                        </span>
                      </div>
                    </div>
                  )}

                  {/* SCROLLABLE SECONDARY DETAILS (VLM Defect Analysis, Model Info, Operator Feedback) */}
                  <div ref={resultPanelScrollRef} className="flex-1 p-3.5 overflow-y-auto space-y-3.5 text-xs">
                    {!currentItem ? (
                      <div className="h-full flex items-center justify-center text-center text-industrial-400 font-mono text-xs">
                        Select a test image from the queue.
                      </div>
                    ) : currentItem.status === 'error' ? (
                      <div className="p-5 bg-reject-50 border border-reject-200 rounded-xl space-y-4">
                        <div className="text-center space-y-2">
                          <div className="w-12 h-12 rounded-2xl bg-reject-100 border border-reject-200 text-reject-600 flex items-center justify-center mx-auto shadow-xs">
                            <AlertTriangle className="w-6 h-6 stroke-[2.2]" />
                          </div>
                          <h2 className="text-base font-extrabold text-reject-900 tracking-tight font-mono uppercase">
                            Inspection Failed
                          </h2>
                          <div className="bg-white p-3 rounded-lg border border-reject-200 text-left">
                            <p className="text-[11px] font-bold text-reject-900 font-mono uppercase mb-0.5">Error Detail:</p>
                            <p className="text-xs text-reject-700 font-mono leading-relaxed">
                              {currentItem.error || (currentItem.result as any)?.error || 'An error occurred during anomaly inference.'}
                            </p>
                          </div>
                        </div>

                        {/* Diagnostic Metadata Box */}
                        <div className="bg-white p-3 rounded-lg border border-reject-200 space-y-2 font-mono text-xs">
                          <div className="flex justify-between items-center text-industrial-600">
                            <span className="text-industrial-500">Target Model:</span>
                            <span className="font-bold text-industrial-900 truncate max-w-[150px]">{selectedModel?.name}</span>
                          </div>
                          <div className="flex justify-between items-center text-industrial-600">
                            <span className="text-industrial-500">File Name:</span>
                            <span className="font-bold text-industrial-900 truncate max-w-[150px]">{currentItem.file.name}</span>
                          </div>
                          <div className="flex justify-between items-center text-industrial-600">
                            <span className="text-industrial-500">Mode:</span>
                            <span className="font-bold text-brand-700 capitalize">
                              {inspectionMode === 'multi_instance' ? 'Multi-Product (Gemini)' : 'Single Product (PatchCore)'}
                            </span>
                          </div>
                        </div>

                        {/* Troubleshooting guidance */}
                        <div className="p-3 bg-reject-100/60 rounded-lg text-[11px] text-reject-800 space-y-1 font-sans">
                          <p className="font-bold">Suggested Action:</p>
                          <p className="leading-snug text-reject-700">
                            Check that model is in <strong className="font-semibold text-reject-900">ACTIVE</strong> status with a trained coreset, and retry this image.
                          </p>
                        </div>

                        {/* Action Buttons */}
                        <div className="space-y-2 pt-1">
                          <Button
                            type="button"
                            variant="primary"
                            size="sm"
                            className="w-full justify-center bg-reject-600 hover:bg-reject-700 text-white"
                            onClick={() => handleRetrySingleItem(selectedIndex)}
                            icon={<RefreshCw className="w-3.5 h-3.5" />}
                          >
                            Retry This Image
                          </Button>
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            className="w-full justify-center text-xs"
                            onClick={() => setStage('setup')}
                          >
                            Return to Setup
                          </Button>
                        </div>
                      </div>
                    ) : currentItem.status === 'processing' || !currentItem.result ? (
                      <>
                        {/* Loading Status Banner */}
                        <div className="p-4 bg-brand-50 border border-brand-200 rounded-lg text-center space-y-2">
                          <div className="w-8 h-8 border-3 border-brand-500 border-t-transparent rounded-full animate-spin mx-auto" />
                          <h2 className="text-lg font-extrabold text-brand-700 tracking-tight font-mono uppercase animate-pulse">
                            ANALYZING
                          </h2>
                          <p className="text-[11px] font-semibold text-brand-600 font-mono">
                            Running PatchCore Feature Comparison...
                          </p>
                        </div>

                        {/* Model Metadata */}
                        <div className="bg-white p-3 rounded border border-industrial-200 space-y-1 text-industrial-600 font-mono">
                          <div className="flex items-center space-x-1.5 font-semibold text-industrial-900 mb-1">
                            <Layers className="w-3.5 h-3.5 text-brand-600" />
                            <span>{selectedModel?.name || 'Textile Inspection'}</span>
                          </div>
                          <p className="text-[10px] text-industrial-400">ID: {selectedModel?.id || selectedModel?._id || '—'}</p>
                        </div>
                      </>
                    ) : (
                      <>
                        {/* Model Metadata */}
                        <div className="bg-industrial-50 p-3 rounded-lg border border-industrial-200 space-y-1 text-industrial-600 font-mono">
                          <div className="flex items-center space-x-1.5 font-semibold text-industrial-900 mb-0.5">
                            <Layers className="w-3.5 h-3.5 text-brand-600" />
                            <span>{selectedModel?.name}</span>
                          </div>
                          <p className="text-[10px] text-industrial-400">ID: {currentItem.result.model_id}</p>
                        </div>

                        {/* AI Defect Analysis (Gemini Interpretation Layer) */}
                        <VLMDefectAnalysisCard
                          isPass={
                            currentItem.result.prediction?.status === 'normal' ||
                            currentItem.result.prediction?.status === 'PASS' ||
                            (currentItem.result.prediction?.anomaly_score !== undefined &&
                              currentItem.result.prediction?.threshold !== undefined &&
                              currentItem.result.prediction.anomaly_score < currentItem.result.prediction.threshold)
                          }
                          vlmAnalysis={currentItem.result.vlm_analysis}
                          onGenerate={handleGenerateVlm}
                          isGenerating={isRetryingVlm}
                        />

                        {/* Feedback Widget */}
                        <div className="pt-3 border-t border-industrial-200 space-y-2">
                          <p className="font-bold text-industrial-900 text-xs">Is this inspection result correct?</p>

                          {feedbackSubmitted ? (
                            <div className="p-2.5 bg-pass-50 border border-pass-200 text-pass-700 text-xs rounded font-medium flex items-center space-x-2">
                              <Check className="w-4 h-4 text-pass-600 flex-shrink-0" />
                              <span>Feedback submitted.</span>
                            </div>
                          ) : (
                            <form onSubmit={handleFeedbackSubmit} className="space-y-2">
                              <div className="grid grid-cols-3 gap-1">
                                <button
                                  type="button"
                                  onClick={() => setFeedbackType('correct')}
                                  className={`py-1.5 text-[11px] font-medium rounded border transition-colors ${feedbackType === 'correct'
                                      ? 'bg-pass-50 text-pass-700 border-pass-300 font-bold'
                                      : 'bg-industrial-50 text-industrial-600 border-industrial-200'
                                    }`}
                                >
                                  ✓ Correct
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setFeedbackType('false_positive')}
                                  className={`py-1.5 text-[11px] font-medium rounded border transition-colors ${feedbackType === 'false_positive'
                                      ? 'bg-reject-50 text-reject-700 border-reject-300 font-bold'
                                      : 'bg-industrial-50 text-industrial-600 border-industrial-200'
                                    }`}
                                >
                                  False Pos.
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setFeedbackType('wrong_severity')}
                                  className={`py-1.5 text-[11px] font-medium rounded border transition-colors ${feedbackType === 'wrong_severity'
                                      ? 'bg-review-50 text-review-700 border-review-300 font-bold'
                                      : 'bg-industrial-50 text-industrial-600 border-industrial-200'
                                    }`}
                                >
                                  Severity
                                </button>
                              </div>

                              <input
                                type="text"
                                placeholder="Add optional comment..."
                                value={feedbackComment}
                                onChange={(e) => setFeedbackComment(e.target.value)}
                                className="w-full px-2.5 py-1 text-xs border border-industrial-300 rounded focus:ring-1 focus:ring-brand-500 focus:outline-none"
                              />

                              <Button
                                type="submit"
                                variant="secondary"
                                size="sm"
                                className="w-full text-xs"
                                isLoading={isSubmittingFeedback}
                                icon={<Send className="w-3 h-3" />}
                              >
                                Submit Feedback
                              </Button>
                            </form>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Bottom Bar Navigation */}
          <div className="bg-white p-2.5 sm:p-3 rounded-xl border border-industrial-200 shadow-sm flex items-center justify-between shrink-0">
            <Button
              variant="outline"
              size="sm"
              disabled={selectedIndex <= 0}
              onClick={() => {
                setSelectedIndex((prev) => Math.max(0, prev - 1));
                setFeedbackSubmitted(false);
              }}
              icon={<ChevronLeft className="w-4 h-4" />}
            >
              Previous Image
            </Button>

            <span className="text-xs font-mono text-industrial-600 font-medium">
              Image {selectedIndex + 1} of {batchItems.length}
            </span>

            <Button
              variant="outline"
              size="sm"
              disabled={selectedIndex >= batchItems.length - 1}
              onClick={() => {
                setSelectedIndex((prev) => Math.min(batchItems.length - 1, prev + 1));
                setFeedbackSubmitted(false);
              }}
            >
              <span>Next Image</span>
              <ChevronRight className="w-4 h-4 ml-1 inline" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
