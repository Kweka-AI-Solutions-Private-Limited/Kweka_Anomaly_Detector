import React, { useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  MessageSquare,
  Cpu,
  Filter,
  Check,
  Send,
  Eye,
  Layers,
  Search,
  ExternalLink,
  ShieldCheck,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';
import { getModels, getModelVersions } from '../api/models';
import { getInspectionRuns } from '../api/runs';
import { getWorkspaceFeedback, submitExpandedFeedback } from '../api/feedback';
import { getStorageUrl } from '../api/client';
import { Model, ModelVersion, InspectionRun, Feedback } from '../types';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { Button } from '../components/common/Button';
import { LoadingSpinner } from '../components/common/LoadingSpinner';

export const FeedbackPage: React.FC = () => {
  const navigate = useNavigate();

  // Filter options state
  const [models, setModels] = useState<Model[]>([]);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [runs, setRuns] = useState<InspectionRun[]>([]);

  // Filter selection state
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [selectedVersionId, setSelectedVersionId] = useState<string>('');
  const [selectedRunId, setSelectedRunId] = useState<string>('');
  const [selectedFeedbackType, setSelectedFeedbackType] = useState<string>('');

  // Aggregated workspace feedback records
  const [feedbackRecords, setFeedbackRecords] = useState<Feedback[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Active item for side-drawer detail review
  const [activeFeedback, setActiveFeedback] = useState<Feedback | null>(null);

  // Correction update form state inside workspace
  const [detectionFeedback, setDetectionFeedback] = useState<'correct' | 'false_positive' | 'false_negative'>('correct');
  const [vlmCategories, setVlmCategories] = useState<('correct' | 'wrong_defect_type' | 'wrong_location' | 'wrong_severity')[]>(['correct']);
  const [correctedType, setCorrectedType] = useState<string>('');
  const [correctedLoc, setCorrectedLoc] = useState<string>('');
  const [correctedSev, setCorrectedSev] = useState<string>('');
  const [commentText, setCommentText] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [submitSuccess, setSubmitSuccess] = useState<boolean>(false);

  // Load models on mount
  useEffect(() => {
    async function initWorkspace() {
      try {
        setIsLoading(true);
        const modelList = await getModels();
        setModels(modelList);
        if (modelList.length > 0) {
          setSelectedModelId(modelList[0].id || modelList[0]._id!);
        }
      } catch (err) {
        setError('Failed to initialize models for Feedback Workspace.');
      } finally {
        setIsLoading(false);
      }
    }
    initWorkspace();
  }, []);

  // When selectedModelId changes, load model versions and inspection runs
  useEffect(() => {
    async function loadModelDependencies() {
      if (!selectedModelId) return;
      try {
        const [vList, rList] = await Promise.all([
          getModelVersions(selectedModelId),
          getInspectionRuns({ model_id: selectedModelId }),
        ]);
        setVersions(vList);
        setRuns(rList);
        setSelectedVersionId('');
        setSelectedRunId('');
      } catch (err) {
        setVersions([]);
        setRuns([]);
      }
    }
    loadModelDependencies();
  }, [selectedModelId]);

  // Load aggregated workspace feedback based on active filters
  const fetchWorkspaceData = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await getWorkspaceFeedback({
        model_id: selectedModelId || undefined,
        model_version_id: selectedVersionId || undefined,
        run_id: selectedRunId || undefined,
        feedback_type: selectedFeedbackType || undefined,
      });
      setFeedbackRecords(data);
      if (data.length > 0) {
        setActiveFeedback(data[0]);
      } else {
        setActiveFeedback(null);
      }
    } catch (err) {
      setError('Failed to fetch workspace feedback records.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchWorkspaceData();
  }, [selectedModelId, selectedVersionId, selectedRunId, selectedFeedbackType]);

  // Sync side review panel state with active selected feedback record
  useEffect(() => {
    if (activeFeedback) {
      setDetectionFeedback(activeFeedback.detection_feedback || 'correct');
      setVlmCategories(activeFeedback.vlm_feedback_categories || ['correct']);
      setCorrectedType(activeFeedback.corrected_defect_type || '');
      setCorrectedLoc(activeFeedback.corrected_location || '');
      setCorrectedSev(activeFeedback.corrected_severity || '');
      setCommentText(activeFeedback.comment || '');
      setSubmitSuccess(false);
    }
  }, [activeFeedback]);

  const handleUpdateFeedback = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeFeedback || !activeFeedback.inspection_id) return;

    try {
      setIsSubmitting(true);
      const updated = await submitExpandedFeedback({
        inspectionId: activeFeedback.inspection_id,
        detection_feedback: detectionFeedback,
        vlm_feedback_categories: vlmCategories,
        corrected_defect_type: correctedType || null,
        corrected_location: correctedLoc || null,
        corrected_severity: correctedSev || null,
        comment: commentText || null,
      });

      // Update workspace list
      setFeedbackRecords((prev) =>
        prev.map((item) => ((item.id || item._id) === (updated.id || updated._id) ? { ...item, ...updated } : item))
      );
      setActiveFeedback(updated);
      setSubmitSuccess(true);
      setTimeout(() => setSubmitSuccess(false), 4000);
    } catch (err: any) {
      setError('Failed to update operator feedback record.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Workspace Header */}
      <div className="bg-white p-6 rounded-lg border border-industrial-200 shadow-sm flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold text-industrial-900 tracking-tight">Feedback Review Workspace</h1>
            <span className="px-2 py-0.5 bg-brand-50 text-brand-700 border border-brand-200 rounded text-[10px] font-mono font-bold uppercase">
              Model Improvement Audit
            </span>
          </div>
          <p className="text-sm text-industrial-500 mt-1">
            Aggregated operator corrections linked to Model → Version → Run → Inspection hierarchy for future retraining cycles.
          </p>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={fetchWorkspaceData}
          icon={<RefreshCw className="w-3.5 h-3.5" />}
        >
          Refresh Data
        </Button>
      </div>

      {/* Filters Bar */}
      <Card className="p-4 bg-industrial-50 border-industrial-200">
        <div className="flex items-center space-x-2 mb-3">
          <Filter className="w-4 h-4 text-brand-600" />
          <span className="text-xs font-mono font-bold text-industrial-800 uppercase tracking-wider">
            Workspace Hierarchical Filters
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          {/* Model Filter */}
          <div>
            <label className="block text-[10px] font-mono text-industrial-500 uppercase mb-1">Model:</label>
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full px-2.5 py-1.5 bg-white border border-industrial-300 rounded font-mono focus:ring-1 focus:ring-brand-500 focus:outline-none"
            >
              <option value="">All Models</option>
              {models.map((m) => (
                <option key={m.id || m._id} value={m.id || m._id}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>

          {/* Model Version Filter */}
          <div>
            <label className="block text-[10px] font-mono text-industrial-500 uppercase mb-1">Version:</label>
            <select
              value={selectedVersionId}
              onChange={(e) => setSelectedVersionId(e.target.value)}
              className="w-full px-2.5 py-1.5 bg-white border border-industrial-300 rounded font-mono focus:ring-1 focus:ring-brand-500 focus:outline-none"
            >
              <option value="">All Versions</option>
              {versions.map((v) => (
                <option key={v.id || v._id} value={v.id || v._id}>
                  v{v.version_number} ({v.status})
                </option>
              ))}
            </select>
          </div>

          {/* Inspection Run Filter */}
          <div>
            <label className="block text-[10px] font-mono text-industrial-500 uppercase mb-1">Inspection Run:</label>
            <select
              value={selectedRunId}
              onChange={(e) => setSelectedRunId(e.target.value)}
              className="w-full px-2.5 py-1.5 bg-white border border-industrial-300 rounded font-mono focus:ring-1 focus:ring-brand-500 focus:outline-none"
            >
              <option value="">All Runs</option>
              {runs.map((r) => (
                <option key={r.id || r._id} value={r.id || r._id}>
                  Run #{r.run_number} ({r.completed_images} images)
                </option>
              ))}
            </select>
          </div>

          {/* Feedback Type Filter */}
          <div>
            <label className="block text-[10px] font-mono text-industrial-500 uppercase mb-1">Feedback Category:</label>
            <select
              value={selectedFeedbackType}
              onChange={(e) => setSelectedFeedbackType(e.target.value)}
              className="w-full px-2.5 py-1.5 bg-white border border-industrial-300 rounded font-mono focus:ring-1 focus:ring-brand-500 focus:outline-none"
            >
              <option value="">All Categories</option>
              <option value="correct">✓ Correct</option>
              <option value="false_positive">False Positive</option>
              <option value="false_negative">False Negative</option>
              <option value="wrong_defect_type">Wrong Defect Type</option>
              <option value="wrong_location">Wrong Location</option>
              <option value="wrong_severity">Wrong Severity</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Main Workspace split view */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Feedback Records Audit List (7 Cols) */}
        <Card className="lg:col-span-7 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-industrial-700">
              Feedback Audit Records ({feedbackRecords.length})
            </h2>
            <span className="text-xs font-mono text-industrial-400">Non-destructive dataset</span>
          </div>

          {isLoading ? (
            <LoadingSpinner label="Loading feedback records..." size="md" />
          ) : feedbackRecords.length === 0 ? (
            <div className="p-12 text-center text-industrial-400 font-mono text-xs space-y-2">
              <MessageSquare className="w-8 h-8 text-industrial-300 mx-auto" />
              <p>No operator feedback entries found matching active filters.</p>
            </div>
          ) : (
            <div className="overflow-x-auto border border-industrial-200 rounded-md max-h-[560px]">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-industrial-50 text-industrial-600 border-b border-industrial-200 uppercase sticky top-0">
                  <tr>
                    <th className="py-2.5 px-3">Sample</th>
                    <th className="py-2.5 px-3">PatchCore Status</th>
                    <th className="py-2.5 px-3">Detection FB</th>
                    <th className="py-2.5 px-3">VLM Categories</th>
                    <th className="py-2.5 px-3">Date</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-industrial-200 bg-white">
                  {feedbackRecords.map((fb) => {
                    const isSelected = (fb.id || fb._id) === (activeFeedback?.id || activeFeedback?._id);
                    const sampleUrl = getStorageUrl(fb.storage_uri);

                    return (
                      <tr
                        key={fb.id || fb._id}
                        onClick={() => setActiveFeedback(fb)}
                        className={`cursor-pointer transition-colors ${
                          isSelected ? 'bg-brand-50/70 font-semibold' : 'hover:bg-industrial-50'
                        }`}
                      >
                        <td className="py-2.5 px-3">
                          <div className="flex items-center space-x-2">
                            {sampleUrl ? (
                              <img
                                src={sampleUrl}
                                alt="Sample"
                                className="w-8 h-8 object-cover rounded border border-industrial-200 bg-industrial-100"
                              />
                            ) : (
                              <div className="w-8 h-8 rounded bg-industrial-100 border border-industrial-200 flex items-center justify-center text-[9px] text-industrial-400">
                                N/A
                              </div>
                            )}
                            <span className="truncate max-w-[110px] text-industrial-900">{fb.filename || 'sample.png'}</span>
                          </div>
                        </td>
                        <td className="py-2.5 px-3">
                          <Badge status={fb.prediction?.status || 'normal'} size="sm" />
                        </td>
                        <td className="py-2.5 px-3">
                          <span className={`uppercase text-[10px] font-bold px-1.5 py-0.5 rounded border ${
                            fb.detection_feedback === 'correct'
                              ? 'bg-pass-50 text-pass-700 border-pass-200'
                              : fb.detection_feedback === 'false_positive'
                              ? 'bg-reject-50 text-reject-700 border-reject-200'
                              : 'bg-review-50 text-review-700 border-review-200'
                          }`}>
                            {fb.detection_feedback || fb.feedback_type}
                          </span>
                        </td>
                        <td className="py-2.5 px-3">
                          <div className="flex flex-wrap gap-1">
                            {(fb.vlm_feedback_categories || []).map((c) => (
                              <span key={c} className="text-[9px] px-1 bg-brand-50 text-brand-700 border border-brand-200 rounded">
                                {c}
                              </span>
                            ))}
                          </div>
                        </td>
                        <td className="py-2.5 px-3 text-[10px] text-industrial-500">
                          {fb.created_at ? new Date(fb.created_at).toLocaleDateString() : 'Just now'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        {/* Right: Detailed Audit Record Inspector (5 Cols) */}
        <Card className="lg:col-span-5 p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-industrial-200 pb-3">
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-industrial-800 flex items-center space-x-1.5">
              <Eye className="w-4 h-4 text-brand-600" />
              <span>Feedback Detail Audit</span>
            </h2>
            {activeFeedback?.inspection_id && (
              <Link
                to={`/inspections/${activeFeedback.inspection_id}`}
                className="text-[11px] font-mono text-brand-600 hover:underline flex items-center space-x-1"
              >
                <span>View Full Report</span>
                <ExternalLink className="w-3 h-3" />
              </Link>
            )}
          </div>

          {!activeFeedback ? (
            <div className="h-48 flex items-center justify-center text-center text-industrial-400 font-mono text-xs">
              Select a feedback entry to view audit details and edit corrections.
            </div>
          ) : (
            <div className="space-y-4 text-xs font-mono">
              {/* Sample preview card */}
              <div className="bg-industrial-900 p-2 rounded-lg border border-industrial-300 h-40 flex items-center justify-center">
                {getStorageUrl(activeFeedback.storage_uri) ? (
                  <img
                    src={getStorageUrl(activeFeedback.storage_uri)}
                    alt="Sample"
                    className="max-h-full max-w-full object-contain rounded"
                  />
                ) : (
                  <span className="text-xs text-industrial-500">No Image Available</span>
                )}
              </div>

              {/* Hierarchy Context readout */}
              <div className="bg-industrial-50 p-3 rounded-lg border border-industrial-200 space-y-1 text-[11px]">
                <div className="flex justify-between">
                  <span className="text-industrial-500">Model ID:</span>
                  <span className="font-bold text-industrial-900 truncate max-w-[160px]">{activeFeedback.model_id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-industrial-500">Model Version:</span>
                  <span className="font-bold text-industrial-900 truncate max-w-[160px]">{activeFeedback.model_version_id}</span>
                </div>
                {activeFeedback.run_id && (
                  <div className="flex justify-between">
                    <span className="text-industrial-500">Inspection Run:</span>
                    <span className="font-bold text-brand-700 truncate max-w-[160px]">{activeFeedback.run_id}</span>
                  </div>
                )}
              </div>

              {/* Original Predictions Snapshot readout */}
              <div className="p-3 bg-white border border-industrial-200 rounded-lg space-y-1 text-[11px]">
                <span className="text-[10px] font-bold text-industrial-400 uppercase">Original Model Verdict Snapshot:</span>
                <div className="flex justify-between items-center">
                  <span>PatchCore Status:</span>
                  <Badge status={activeFeedback.original_prediction?.status || activeFeedback.prediction?.status || 'normal'} size="sm" />
                </div>
                {activeFeedback.original_vlm_analysis?.defect_type && (
                  <div className="flex justify-between">
                    <span>VLM Defect:</span>
                    <span className="font-bold text-industrial-800">{activeFeedback.original_vlm_analysis.defect_type}</span>
                  </div>
                )}
              </div>

              {/* Update Feedback Form inside Workspace */}
              <form onSubmit={handleUpdateFeedback} className="space-y-3 pt-2 border-t border-industrial-200">
                <span className="text-[11px] font-bold text-industrial-800 uppercase">Modify Saved Operator Feedback:</span>

                {submitSuccess && (
                  <div className="p-2.5 bg-pass-50 border border-pass-200 text-pass-700 text-[11px] rounded flex items-center space-x-2 font-sans">
                    <Check className="w-3.5 h-3.5 text-pass-600" />
                    <span>Feedback audit entry updated!</span>
                  </div>
                )}

                <div className="space-y-1">
                  <label className="text-[10px] text-industrial-500 uppercase">Detection Verdict:</label>
                  <div className="grid grid-cols-2 gap-1.5">
                    <button
                      type="button"
                      onClick={() => setDetectionFeedback('correct')}
                      className={`py-1.5 px-2 text-[10px] font-bold rounded border ${
                        detectionFeedback === 'correct' ? 'bg-pass-50 text-pass-700 border-pass-300' : 'bg-industrial-50 text-industrial-600 border-industrial-200'
                      }`}
                    >
                      ✓ Correct
                    </button>
                    <button
                      type="button"
                      onClick={() => setDetectionFeedback('false_positive')}
                      className={`py-1.5 px-2 text-[10px] font-bold rounded border ${
                        detectionFeedback === 'false_positive' ? 'bg-reject-50 text-reject-700 border-reject-300' : 'bg-industrial-50 text-industrial-600 border-industrial-200'
                      }`}
                    >
                      False Positive
                    </button>
                  </div>
                </div>

                <div className="space-y-1">
                  <label className="text-[10px] text-industrial-500 uppercase">Operator Comment:</label>
                  <textarea
                    rows={2}
                    value={commentText}
                    onChange={(e) => setCommentText(e.target.value)}
                    className="w-full p-2 text-xs border border-industrial-300 rounded font-sans focus:ring-1 focus:ring-brand-500 focus:outline-none"
                  />
                </div>

                <Button
                  type="submit"
                  variant="primary"
                  size="sm"
                  className="w-full"
                  isLoading={isSubmitting}
                  icon={<Send className="w-3 h-3" />}
                >
                  Save Corrections
                </Button>
              </form>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};
