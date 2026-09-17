import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
  Cpu, ArrowLeft, Layers, PlayCircle, Images, Calendar, CheckCircle2,
  AlertTriangle, Power, Clock, ChevronRight, Activity, Eye, RefreshCw,
  Info, Trash2
} from 'lucide-react';
import {
  getModel, getModelVersions, activateModel, deactivateModel,
  activateModelVersion, deleteModelVersion, deleteModel
} from '../api/models';
import { getInspectionRuns, getInspectionRun } from '../api/runs';
import { getInspections } from '../api/inspections';
import { Model, ModelVersion, InspectionRun, Inspection } from '../types';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { Button } from '../components/common/Button';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { EmptyState } from '../components/common/EmptyState';
import { ReferenceImagesModal } from '../components/models/ReferenceImagesModal';

export const ModelDetails: React.FC = () => {
  const { modelId } = useParams<{ modelId: string }>();
  const navigate = useNavigate();

  const [model, setModel] = useState<Model | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [runs, setRuns] = useState<InspectionRun[]>([]);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<InspectionRun | null>(null);
  const [runInspections, setRunInspections] = useState<Inspection[]>([]);
  const [viewingRefVersion, setViewingRefVersion] = useState<ModelVersion | null>(null);

  const [isLoadingModel, setIsLoadingModel] = useState<boolean>(true);
  const [isLoadingRuns, setIsLoadingRuns] = useState<boolean>(false);
  const [isLoadingRunDetail, setIsLoadingRunDetail] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Deletion Modal States
  const [versionToDelete, setVersionToDelete] = useState<ModelVersion | null>(null);
  const [isDeletingVersion, setIsDeletingVersion] = useState<boolean>(false);
  const [deleteVersionError, setDeleteVersionError] = useState<string | null>(null);

  const [showDeleteModelModal, setShowDeleteModelModal] = useState<boolean>(false);
  const [deleteModelInputName, setDeleteModelInputName] = useState<string>('');
  const [isDeletingModel, setIsDeletingModel] = useState<boolean>(false);
  const [deleteModelError, setDeleteModelError] = useState<string | null>(null);

  const activeVersion = versions.find(v => (v.id || v._id) === model?.active_version_id) || versions[0];

  const handleActivateVersion = async (versionId: string) => {
    if (!modelId) return;
    try {
      setError(null);
      await activateModelVersion(modelId, versionId);
      const [modelData, versionsData] = await Promise.all([
        getModel(modelId),
        getModelVersions(modelId)
      ]);
      setModel(modelData);
      setVersions(versionsData);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to activate version.');
    }
  };

  const handleDeleteVersionConfirm = async () => {
    if (!modelId || !versionToDelete) return;
    const vId = versionToDelete.id || versionToDelete._id || '';
    try {
      setIsDeletingVersion(true);
      setDeleteVersionError(null);
      await deleteModelVersion(modelId, vId);

      setVersionToDelete(null);
      if (selectedVersionId === vId) {
        setSelectedVersionId(null);
      }

      const [modelData, versionsData] = await Promise.all([
        getModel(modelId),
        getModelVersions(modelId)
      ]);
      setModel(modelData);
      setVersions(versionsData);
    } catch (err: any) {
      setDeleteVersionError(err.response?.data?.detail || err.message || 'Failed to delete model version.');
    } finally {
      setIsDeletingVersion(false);
    }
  };

  const handleDeleteModelConfirm = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!modelId || !model) return;
    if (deleteModelInputName.trim() !== model.name.trim()) {
      setDeleteModelError(`Please type "${model.name}" exactly to confirm deletion.`);
      return;
    }

    try {
      setIsDeletingModel(true);
      setDeleteModelError(null);
      await deleteModel(modelId);
      setShowDeleteModelModal(false);
      navigate('/models');
    } catch (err: any) {
      setDeleteModelError(err.response?.data?.detail || err.message || 'Failed to delete model.');
    } finally {
      setIsDeletingModel(false);
    }
  };

  // Load Model & Versions
  useEffect(() => {
    if (!modelId) return;

    async function loadData() {
      try {
        setIsLoadingModel(true);
        setError(null);
        const [modelData, versionsData] = await Promise.all([
          getModel(modelId!),
          getModelVersions(modelId!)
        ]);
        setModel(modelData);
        setVersions(versionsData);
      } catch (err: any) {
        setError(err.response?.data?.detail || err.message || 'Failed to load model details.');
      } finally {
        setIsLoadingModel(false);
      }
    }

    loadData();
  }, [modelId]);

  // Load Runs when modelId or selectedVersionId changes
  useEffect(() => {
    if (!modelId) return;

    async function loadRuns() {
      try {
        setIsLoadingRuns(true);
        const params: { model_id: string; model_version_id?: string } = { model_id: modelId! };
        if (selectedVersionId) {
          params.model_version_id = selectedVersionId;
        }
        const runsData = await getInspectionRuns(params);
        setRuns(runsData);
      } catch (err: any) {
        console.error('Failed to load runs:', err);
      } finally {
        setIsLoadingRuns(false);
      }
    }

    loadRuns();
  }, [modelId, selectedVersionId]);

  // Load Selected Run Detail & Inspections when selectedRunId changes
  useEffect(() => {
    if (!selectedRunId) {
      setSelectedRun(null);
      setRunInspections([]);
      return;
    }

    async function loadRunDetail() {
      try {
        setIsLoadingRunDetail(true);
        const [runData, inspList] = await Promise.all([
          getInspectionRun(selectedRunId!),
          getInspections({ run_id: selectedRunId! })
        ]);
        setSelectedRun(runData);
        setRunInspections(inspList);
      } catch (err: any) {
        console.error('Failed to load run detail:', err);
      } finally {
        setIsLoadingRunDetail(false);
      }
    }

    loadRunDetail();
  }, [selectedRunId]);

  if (isLoadingModel) {
    return <LoadingSpinner label="Loading model details..." size="lg" />;
  }

  if (error || !model) {
    return (
      <div className="space-y-6">
        <Button variant="outline" size="sm" onClick={() => navigate('/models')} icon={<ArrowLeft className="w-4 h-4" />}>
          Back to Models
        </Button>
        <div className="p-6 bg-reject-50 border border-reject-200 text-reject-700 rounded-xl flex items-center space-x-3">
          <AlertTriangle className="w-5 h-5 flex-shrink-0" />
          <span>{error || 'Model not found.'}</span>
        </div>
      </div>
    );
  }

  const selectedVersion = versions.find(v => (v.id || v._id) === selectedVersionId);
  const totalInspectedImages = runs.reduce((acc, r) => acc + (r.total_images || 0), 0);

  return (
    <div className="space-y-8">
      {/* Contextual Breadcrumb */}
      <nav className="flex items-center space-x-2 text-sm font-medium text-industrial-500">
        <Link to="/models" className="hover:text-brand-600 transition-colors">
          Models
        </Link>
        <ChevronRight className="w-4 h-4 text-industrial-400" />
        <span className="text-industrial-900 font-bold">{model.name}</span>

        {selectedVersion && (
          <>
            <ChevronRight className="w-4 h-4 text-industrial-400" />
            <span className="text-industrial-700">Version #{selectedVersion.version_number}</span>
          </>
        )}

        {selectedRun && (
          <>
            <ChevronRight className="w-4 h-4 text-industrial-400" />
            <span className="text-brand-700 font-bold">Run #{selectedRun.run_number}</span>
          </>
        )}
      </nav>

      {/* Model Header Card */}
      <div className="bg-white p-7 rounded-2xl border border-industrial-200 shadow-sm space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex items-center space-x-4">
            <div className="p-4 bg-brand-50 rounded-2xl text-brand-600 border border-brand-200">
              <Cpu className="w-8 h-8" />
            </div>
            <div>
              <div className="flex items-center space-x-3">
                <h1 className="text-2xl sm:text-3xl font-black text-industrial-900 tracking-tight">
                  {model.name}
                </h1>
                <Badge status={model.status} />
              </div>
              <p className="text-sm text-industrial-500 font-mono mt-1">
                Model ID: {model.id || model._id}
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => navigate('/models')}
              icon={<ArrowLeft className="w-4 h-4" />}
            >
              Back to Models
            </Button>
            {model.status === 'active' && (
              <Button
                variant="primary"
                size="sm"
                onClick={() => navigate(`/inspect?model_id=${model.id || model._id}`)}
                icon={<PlayCircle className="w-4 h-4" />}
              >
                Run New Inspection
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setShowDeleteModelModal(true);
                setDeleteModelInputName('');
                setDeleteModelError(null);
              }}
              icon={<Trash2 className="w-4 h-4 text-rose-600" />}
              className="border-rose-200 text-rose-700 hover:bg-rose-50 hover:border-rose-300"
            >
              Delete Model
            </Button>
          </div>
        </div>

        {model.description && (
          <p className="text-base text-industrial-600 leading-relaxed border-t border-industrial-100 pt-4">
            {model.description}
          </p>
        )}

        {/* Model Metrics Summary Row */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 pt-2">
          <div className="bg-industrial-50 p-4 rounded-xl border border-industrial-200">
            <span className="text-xs text-industrial-500 font-mono block">VERSIONS</span>
            <span className="text-xl font-bold text-industrial-900">{versions.length}</span>
          </div>
          <div className="bg-industrial-50 p-4 rounded-xl border border-industrial-200">
            <span className="text-xs text-industrial-500 font-mono block">INSPECTION RUNS</span>
            <span className="text-xl font-bold text-industrial-900">{runs.length}</span>
          </div>
          <div className="bg-industrial-50 p-4 rounded-xl border border-industrial-200">
            <span className="text-xs text-industrial-500 font-mono block">TOTAL IMAGES</span>
            <span className="text-xl font-bold text-industrial-900">{totalInspectedImages}</span>
          </div>
          <div className="bg-industrial-50 p-4 rounded-xl border border-industrial-200">
            <span className="text-xs text-industrial-500 font-mono block">GOOD REFERENCES</span>
            <span className="text-xl font-bold text-industrial-900">{model.reference_image_count}</span>
          </div>
          <div className="bg-industrial-50 p-4 rounded-xl border border-industrial-200">
            <span className="text-xs text-industrial-500 font-mono block">CREATED DATE</span>
            <span className="text-sm font-bold text-industrial-900 mt-1 block">
              {new Date(model.created_at).toLocaleDateString()}
            </span>
          </div>
        </div>
      </div>



      {/* SECTION 1: MODEL VERSIONS */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-extrabold text-industrial-900 flex items-center space-x-2">
            <Layers className="w-5 h-5 text-brand-600" />
            <span>Model Versions</span>
          </h2>
          {selectedVersionId && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSelectedVersionId(null)}
            >
              Show All Versions ({versions.length})
            </Button>
          )}
        </div>

        {versions.length === 0 ? (
          <EmptyState
            icon={Layers}
            title="No Model Versions"
            description="This model has not been built into a usable model version yet."
            actionLabel="Build Model"
            onAction={() => navigate(`/models/${model.id || model._id}/build`)}
          />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {versions.map((ver) => {
              const vId = ver.id || ver._id || '';
              const isActiveVersion = model.active_version_id === vId;
              const isSelected = selectedVersionId === vId;

              return (
                <Card
                  key={vId}
                  className={`p-6 space-y-4 border transition-all ${
                    isSelected
                      ? 'border-brand-500 bg-brand-50/20 ring-2 ring-brand-500/20 shadow-md'
                      : 'border-industrial-200 bg-white hover:border-industrial-300'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="font-extrabold text-lg text-industrial-900">
                        Version #{ver.version_number}
                      </span>
                      {isActiveVersion && (
                        <Badge status="active">ACTIVE VERSION</Badge>
                      )}
                    </div>
                    <span className="text-xs text-industrial-400 font-mono">
                      ID: {vId.slice(-6)}
                    </span>
                  </div>

                  <div className="space-y-2 text-sm text-industrial-600 font-mono bg-industrial-50 p-3 rounded-lg border border-industrial-100">
                    <div className="flex justify-between">
                      <span className="text-industrial-500">Built Date:</span>
                      <span className="font-semibold text-industrial-900">
                        {new Date(ver.created_at).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-industrial-500">GOOD References:</span>
                      <span className="font-semibold text-industrial-900">
                        {ver.training?.reference_count ?? model.reference_image_count}
                      </span>
                    </div>
                    {ver.calibration?.threshold !== undefined && (
                      <div className="flex justify-between">
                        <span className="text-industrial-500">Calibrated Threshold:</span>
                        <span className="font-bold text-brand-700">
                          {ver.calibration.threshold.toFixed(2)}
                        </span>
                      </div>
                    )}
                  </div>

                  <div className="pt-2 flex flex-wrap items-center gap-2">
                    {!isActiveVersion && (
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => handleActivateVersion(vId)}
                        icon={<CheckCircle2 className="w-4 h-4 text-emerald-600" />}
                      >
                        Activate Version
                      </Button>
                    )}
                    <Button
                      variant={isSelected ? 'primary' : 'outline'}
                      size="sm"
                      onClick={() => {
                        setSelectedVersionId(isSelected ? null : vId);
                        setSelectedRunId(null);
                      }}
                      icon={<Activity className="w-4 h-4" />}
                    >
                      {isSelected ? 'Viewing Runs' : 'View Runs'}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setViewingRefVersion(ver)}
                      icon={<Images className="w-4 h-4" />}
                    >
                      View Reference Images
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={isActiveVersion && model.status === 'active'}
                      title={
                        isActiveVersion && model.status === 'active'
                          ? 'Cannot delete the active version of an active model. Activate another version or deactivate the model first.'
                          : 'Delete model version'
                      }
                      onClick={() => {
                        setVersionToDelete(ver);
                        setDeleteVersionError(null);
                      }}
                      icon={<Trash2 className="w-4 h-4 text-rose-600" />}
                      className="border-rose-200 text-rose-700 hover:bg-rose-50 hover:border-rose-300 disabled:opacity-50"
                    >
                      Delete
                    </Button>
                  </div>
                </Card>
              );
            })}
          </div>
        )}
      </div>

      {/* SECTION 2: INSPECTION RUNS */}
      <div className="space-y-4 pt-4 border-t border-industrial-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <h2 className="text-xl font-extrabold text-industrial-900 flex items-center space-x-2">
              <PlayCircle className="w-5 h-5 text-brand-600" />
              <span>Inspection Runs</span>
            </h2>
            {selectedVersion && (
              <Badge type="info">
                Version #{selectedVersion.version_number}
              </Badge>
            )}
          </div>
          <span className="text-sm font-mono text-industrial-500">
            Total Runs: {runs.length}
          </span>
        </div>

        {isLoadingRuns ? (
          <LoadingSpinner label="Loading inspection runs..." size="md" />
        ) : runs.length === 0 ? (
          <EmptyState
            icon={PlayCircle}
            title="No Inspection Runs Found"
            description={
              selectedVersion
                ? `No inspection runs recorded under Version #${selectedVersion.version_number} yet.`
                : 'No inspection runs executed under this model yet.'
            }
          />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {runs.map((run) => {
              const rId = run.id || run._id || run.run_id || '';
              const isSelected = selectedRunId === rId;
              const passCount = run.pass_count ?? run.summary?.pass ?? 0;
              const rejectCount = run.reject_count ?? run.summary?.reject ?? 0;
              const errorCount = run.error_count ?? run.summary?.errors ?? 0;
              const runVer = versions.find(v => (v.id || v._id) === run.model_version_id);

              return (
                <Card
                  key={rId}
                  className={`p-6 space-y-4 border transition-all cursor-pointer ${
                    isSelected
                      ? 'border-brand-500 bg-brand-50/20 ring-2 ring-brand-500/20 shadow-md'
                      : 'border-industrial-200 bg-white hover:border-brand-300 hover:shadow-sm'
                  }`}
                  onClick={() => setSelectedRunId(isSelected ? null : rId)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="font-black text-xl text-industrial-900">
                        Run #{run.run_number}
                      </span>
                      <Badge status={run.status} />
                    </div>
                    <span className="text-xs text-industrial-500 font-mono flex items-center space-x-1">
                      <Calendar className="w-3.5 h-3.5" />
                      <span>{new Date(run.created_at).toLocaleDateString()}</span>
                    </span>
                  </div>

                  <div className="text-xs font-mono text-industrial-500 flex items-center justify-between border-b border-industrial-100 pb-2">
                    <span>Target Model Version:</span>
                    <span className="font-bold text-industrial-800">
                      {runVer ? `Version #${runVer.version_number}` : `V:${run.model_version_id.slice(-6)}`}
                    </span>
                  </div>

                  {/* Summary Metric Chips */}
                  <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono">
                    <div className="p-2 bg-industrial-50 rounded-lg border border-industrial-200">
                      <span className="text-industrial-500 block text-[10px]">TOTAL</span>
                      <span className="font-bold text-industrial-900 text-sm">{run.total_images}</span>
                    </div>
                    <div className="p-2 bg-emerald-50 rounded-lg border border-emerald-200">
                      <span className="text-emerald-600 block text-[10px]">PASS</span>
                      <span className="font-bold text-emerald-800 text-sm">{passCount}</span>
                    </div>
                    <div className="p-2 bg-rose-50 rounded-lg border border-rose-200">
                      <span className="text-rose-600 block text-[10px]">REJECT</span>
                      <span className="font-bold text-rose-800 text-sm">{rejectCount}</span>
                    </div>
                  </div>

                  <div className="pt-2 flex items-center justify-between">
                    <Button
                      variant={isSelected ? 'primary' : 'outline'}
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedRunId(isSelected ? null : rId);
                      }}
                      icon={<Eye className="w-4 h-4" />}
                    >
                      {isSelected ? 'Hide Images' : 'View Inspected Images'}
                    </Button>
                  </div>
                </Card>
              );
            })}
          </div>
        )}
      </div>

      {/* SECTION 3: RUN DETAILS & INSPECTED IMAGES GRID */}
      {selectedRunId && (
        <div className="space-y-5 pt-6 border-t-2 border-brand-200">
          {isLoadingRunDetail ? (
            <LoadingSpinner label="Loading run images..." size="md" />
          ) : selectedRun ? (
            <div className="space-y-6">
              {/* Selected Run Header Summary */}
              <div className="bg-industrial-900 text-white p-6 rounded-2xl shadow-lg space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                  <div>
                    <div className="flex items-center space-x-3">
                      <h3 className="text-2xl font-black tracking-tight text-white">
                        Run #{selectedRun.run_number} Details
                      </h3>
                      <Badge status={selectedRun.status} />
                    </div>
                    <p className="text-xs text-industrial-400 font-mono mt-1">
                      Model: {model.name} | Executed on {new Date(selectedRun.created_at).toLocaleString()}
                    </p>
                  </div>

                  <Button
                    variant="outline"
                    size="sm"
                    className="border-industrial-700 text-white hover:bg-industrial-800"
                    onClick={() => setSelectedRunId(null)}
                  >
                    Close Run View
                  </Button>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 font-mono text-sm">
                  <div className="bg-industrial-800/80 p-3 rounded-xl border border-industrial-700">
                    <span className="text-industrial-400 text-xs block">TOTAL IMAGES</span>
                    <span className="text-lg font-bold text-white">{selectedRun.total_images}</span>
                  </div>
                  <div className="bg-emerald-950/60 p-3 rounded-xl border border-emerald-800 text-emerald-300">
                    <span className="text-emerald-400 text-xs block">PASS COUNT</span>
                    <span className="text-lg font-bold">{selectedRun.pass_count ?? selectedRun.summary?.pass ?? 0}</span>
                  </div>
                  <div className="bg-rose-950/60 p-3 rounded-xl border border-rose-800 text-rose-300">
                    <span className="text-rose-400 text-xs block">REJECT COUNT</span>
                    <span className="text-lg font-bold">{selectedRun.reject_count ?? selectedRun.summary?.reject ?? 0}</span>
                  </div>
                  <div className="bg-industrial-800/80 p-3 rounded-xl border border-industrial-700">
                    <span className="text-industrial-400 text-xs block">ERROR COUNT</span>
                    <span className="text-lg font-bold text-white">{selectedRun.error_count ?? selectedRun.summary?.errors ?? 0}</span>
                  </div>
                </div>
              </div>

              {/* Inspected Image Grid */}
              <div className="space-y-4">
                <h4 className="text-lg font-extrabold text-industrial-900">
                  Inspected Images in Run #{selectedRun.run_number} ({runInspections.length})
                </h4>

                {runInspections.length === 0 ? (
                  <div className="p-8 text-center text-industrial-500 bg-white rounded-2xl border border-industrial-200 font-mono text-sm">
                    No individual inspection records found for this run.
                  </div>
                ) : (
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-5">
                    {runInspections.map((insp) => {
                      const inspId = insp.id || insp._id || insp.inspection_id || '';
                      const predStatus = insp.prediction?.status || 'unknown';
                      const isPass = predStatus === 'normal' || predStatus === 'PASS';
                      const filename = insp.filename || insp.input?.filename || 'sample_image.png';
                      const score = insp.prediction?.anomaly_score;
                      const threshold = insp.prediction?.threshold;
                      const severity = insp.prediction?.severity;

                      return (
                        <Card
                          key={inspId}
                          className="p-4 flex flex-col justify-between space-y-3 hover:border-brand-400 hover:shadow-md transition-all bg-white border-industrial-200 cursor-pointer"
                          onClick={() => navigate(`/inspections/${inspId}`)}
                        >
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-mono font-bold text-industrial-800 truncate max-w-[150px]" title={filename}>
                                {filename}
                              </span>
                              <Badge status={isPass ? 'pass' : 'reject'} />
                            </div>

                            {/* Image Thumbnail Preview */}
                            <div className="relative aspect-video bg-industrial-900 rounded-xl overflow-hidden flex items-center justify-center border border-industrial-200">
                              {insp.storage_uri ? (
                                <img
                                  src={`http://localhost:8000/${insp.storage_uri.replace(/^\//, '')}`}
                                  alt={filename}
                                  className="object-cover w-full h-full"
                                  onError={(e) => {
                                    (e.target as HTMLElement).style.display = 'none';
                                  }}
                                />
                              ) : (
                                <div className="text-industrial-500 flex flex-col items-center">
                                  <Images className="w-6 h-6 mb-1" />
                                  <span className="text-[10px] font-mono">No Image</span>
                                </div>
                              )}
                            </div>

                            <div className="space-y-1 text-xs font-mono text-industrial-600 bg-industrial-50 p-2.5 rounded-lg border border-industrial-100">
                              <div className="flex justify-between">
                                <span className="text-industrial-400">Score:</span>
                                <span className="font-bold text-industrial-900">
                                  {score !== undefined ? score.toFixed(2) : 'N/A'}
                                </span>
                              </div>
                              {threshold !== undefined && (
                                <div className="flex justify-between">
                                  <span className="text-industrial-400">Threshold:</span>
                                  <span className="font-bold text-industrial-700">
                                    {threshold.toFixed(2)}
                                  </span>
                                </div>
                              )}
                              {severity && (
                                <div className="flex justify-between">
                                  <span className="text-industrial-400">Severity:</span>
                                  <span className="font-semibold capitalize text-industrial-800">
                                    {severity}
                                  </span>
                                </div>
                              )}
                            </div>
                          </div>

                          <Button
                            variant="outline"
                            size="sm"
                            className="w-full justify-center"
                            onClick={(e) => {
                              e.stopPropagation();
                              navigate(`/inspections/${inspId}`);
                            }}
                            icon={<Eye className="w-3.5 h-3.5" />}
                          >
                            View Result
                          </Button>
                        </Card>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </div>
      )}

      {viewingRefVersion && model && (
        <ReferenceImagesModal
          modelId={model.id || model._id || ''}
          version={viewingRefVersion}
          onClose={() => setViewingRefVersion(null)}
        />
      )}

      {/* DELETE MODEL VERSION CONFIRMATION MODAL */}
      {versionToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-industrial-950/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white rounded-2xl border border-industrial-200 shadow-2xl w-full max-w-md overflow-hidden p-6 space-y-5">
            <div className="flex items-center space-x-3 text-rose-600">
              <div className="p-3 bg-rose-50 rounded-xl border border-rose-200">
                <Trash2 className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-lg font-extrabold text-industrial-900">
                  Delete Version #{versionToDelete.version_number}?
                </h3>
                <p className="text-xs text-industrial-500 font-mono">
                  Version ID: {(versionToDelete.id || versionToDelete._id || '').slice(-6)}
                </p>
              </div>
            </div>

            <p className="text-sm text-industrial-600 leading-relaxed">
              Are you sure you want to delete <span className="font-bold text-industrial-900 font-mono">Version #{versionToDelete.version_number}</span>?
              Physical ML checkpoint artifacts on disk will be purged. Historical inspection records will be safely preserved.
            </p>

            {deleteVersionError && (
              <div className="p-3 bg-rose-50 border border-rose-200 text-rose-700 rounded-xl text-xs font-mono flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                <span>{deleteVersionError}</span>
              </div>
            )}

            <div className="flex items-center justify-end space-x-3 pt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setVersionToDelete(null)}
                disabled={isDeletingVersion}
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={handleDeleteVersionConfirm}
                disabled={isDeletingVersion}
                icon={isDeletingVersion ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                className="bg-rose-600 hover:bg-rose-700 text-white border-none"
              >
                {isDeletingVersion ? 'Deleting...' : 'Confirm Delete Version'}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* DELETE MODEL CONFIRMATION MODAL */}
      {showDeleteModelModal && model && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-industrial-950/80 backdrop-blur-sm animate-in fade-in duration-200">
          <form
            onSubmit={handleDeleteModelConfirm}
            className="bg-white rounded-2xl border border-industrial-200 shadow-2xl w-full max-w-lg overflow-hidden p-6 space-y-5"
          >
            <div className="flex items-center space-x-3 text-rose-600">
              <div className="p-3 bg-rose-50 rounded-xl border border-rose-200">
                <AlertTriangle className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-xl font-extrabold text-industrial-900">
                  Delete Model '{model.name}'?
                </h3>
                <p className="text-xs text-rose-600 font-mono font-bold">
                  DANGER: Soft-delete model and all associated versions
                </p>
              </div>
            </div>

            <p className="text-sm text-industrial-600 leading-relaxed">
              This action will soft-delete the model and all associated versions from active workflows and purge physical model artifacts from storage. Historical inspection reports and audit trails will remain intact.
            </p>

            <div className="space-y-2 bg-industrial-50 p-4 rounded-xl border border-industrial-200">
              <label className="text-xs font-mono text-industrial-700 block font-bold">
                To confirm deletion, please type <span className="text-rose-600 select-all font-mono font-black">{model.name}</span> below:
              </label>
              <input
                type="text"
                value={deleteModelInputName}
                onChange={(e) => setDeleteModelInputName(e.target.value)}
                placeholder={model.name}
                className="w-full px-3 py-2 bg-white border border-industrial-300 rounded-xl text-sm font-mono text-industrial-900 focus:outline-none focus:ring-2 focus:ring-rose-500/20 focus:border-rose-500"
              />
            </div>

            {deleteModelError && (
              <div className="p-3 bg-rose-50 border border-rose-200 text-rose-700 rounded-xl text-xs font-mono flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                <span>{deleteModelError}</span>
              </div>
            )}

            <div className="flex items-center justify-end space-x-3 pt-2">
              <Button
                variant="outline"
                size="sm"
                type="button"
                onClick={() => setShowDeleteModelModal(false)}
                disabled={isDeletingModel}
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                size="sm"
                type="submit"
                disabled={isDeletingModel || deleteModelInputName.trim() !== model.name.trim()}
                icon={isDeletingModel ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                className="bg-rose-600 hover:bg-rose-700 text-white border-none disabled:opacity-50"
              >
                {isDeletingModel ? 'Deleting...' : 'Delete Entire Model'}
              </Button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};
