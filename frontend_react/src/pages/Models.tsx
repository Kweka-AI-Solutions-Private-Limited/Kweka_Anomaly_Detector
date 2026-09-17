import React, { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Cpu,
  Plus,
  Images,
  Layers,
  ArrowRight,
  AlertTriangle,
  CheckCircle2,
  Power,
  RefreshCw,
  X,
  Folder,
  FolderPlus,
  Tag,
} from 'lucide-react';
import { getModels, activateModel, deactivateModel } from '../api/models';
import { getModelGroups } from '../api/model_groups';
import { Model, ModelGroup } from '../types';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { EmptyState } from '../components/common/EmptyState';
import { ModelGroupManagerModal } from '../components/models/ModelGroupManagerModal';
import { AssignGroupModal } from '../components/models/AssignGroupModal';

export const Models: React.FC = () => {
  const navigate = useNavigate();
  const [models, setModels] = useState<Model[]>([]);
  const [groups, setGroups] = useState<ModelGroup[]>([]);
  const [selectedGroupId, setSelectedGroupId] = useState<string>('all');

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Modals state
  const [isGroupManagerOpen, setIsGroupManagerOpen] = useState<boolean>(false);
  const [assignTargetModel, setAssignTargetModel] = useState<Model | null>(null);

  const [activateTarget, setActivateTarget] = useState<Model | null>(null);
  const [deactivateTarget, setDeactivateTarget] = useState<Model | null>(null);
  const [errorTarget, setErrorTarget] = useState<{ model: Model; message: string } | null>(null);
  const [actionLoadingId, setActionLoadingId] = useState<string | null>(null);

  async function fetchAllData() {
    try {
      setIsLoading(true);
      setError(null);
      const [modelsData, groupsData] = await Promise.all([
        getModels(),
        getModelGroups(),
      ]);
      setModels(modelsData);
      setGroups(groupsData);
    } catch (err: any) {
      setError('Failed to fetch inspection models or groups from backend API.');
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    fetchAllData();
  }, []);

  const handleConfirmActivate = async () => {
    if (!activateTarget) return;
    const modelId = activateTarget.id || activateTarget._id;
    if (!modelId) return;
    try {
      setActionLoadingId(modelId);
      await activateModel(modelId);
      setActivateTarget(null);
      await fetchAllData();
    } catch (err: any) {
      const detail = err.response?.data?.detail || err.message || 'Activation failed.';
      const targetModel = activateTarget;
      setActivateTarget(null);
      setErrorTarget({ model: targetModel, message: detail });
    } finally {
      setActionLoadingId(null);
    }
  };

  const handleConfirmDeactivate = async () => {
    if (!deactivateTarget) return;
    const modelId = deactivateTarget.id || deactivateTarget._id;
    if (!modelId) return;
    try {
      setActionLoadingId(modelId);
      await deactivateModel(modelId);
      setDeactivateTarget(null);
      await fetchAllData();
    } catch (err: any) {
      const detail = err.response?.data?.detail || err.message || 'Deactivation failed.';
      const targetModel = deactivateTarget;
      setDeactivateTarget(null);
      setErrorTarget({ model: targetModel, message: detail });
    } finally {
      setActionLoadingId(null);
    }
  };

  // Filter models based on selected group pill
  const filteredModels = useMemo(() => {
    if (selectedGroupId === 'all') return models;
    if (selectedGroupId === 'ungrouped') return models.filter((m) => !m.group_id);
    return models.filter((m) => m.group_id === selectedGroupId);
  }, [models, selectedGroupId]);

  // Count helper
  const ungroupedCount = useMemo(() => {
    return models.filter((m) => !m.group_id).length;
  }, [models]);

  if (isLoading) {
    return <LoadingSpinner label="Fetching inspection models and groups..." size="lg" />;
  }

  return (
    <div className="space-y-7">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-5 bg-white p-7 sm:p-8 rounded-2xl border border-industrial-200 shadow-sm">
        <div>
          <h1 className="text-3xl font-black text-industrial-900 tracking-tight">Inspection Models</h1>
          <p className="text-base text-industrial-600 mt-1.5 font-normal">
            Manage PatchCore memory-bank anomaly detection models categorized by organizational domain groups.
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <Button
            variant="outline"
            size="md"
            onClick={() => setIsGroupManagerOpen(true)}
            icon={<FolderPlus className="w-5 h-5 text-brand-600" />}
          >
            Manage Groups
          </Button>
          <Button
            variant="primary"
            size="md"
            onClick={() => navigate('/models/create')}
            icon={<Plus className="w-5 h-5" />}
          >
            Create Model
          </Button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-reject-50 border border-reject-200 text-reject-700 text-base rounded-xl flex items-center space-x-3">
          <AlertTriangle className="w-5 h-5 text-reject-600 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Group Filter Pills */}
      <div className="flex items-center space-x-2 overflow-x-auto pb-2 scrollbar-none">
        <button
          onClick={() => setSelectedGroupId('all')}
          className={`px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all flex items-center space-x-2 border whitespace-nowrap cursor-pointer ${
            selectedGroupId === 'all'
              ? 'bg-industrial-900 text-white border-industrial-900 shadow-sm'
              : 'bg-white text-industrial-700 border-industrial-200 hover:bg-industrial-50'
          }`}
        >
          <span>All Models</span>
          <span
            className={`px-2 py-0.5 rounded-full text-[10px] ${
              selectedGroupId === 'all'
                ? 'bg-industrial-700 text-white'
                : 'bg-industrial-100 text-industrial-600'
            }`}
          >
            {models.length}
          </span>
        </button>

        <button
          onClick={() => setSelectedGroupId('ungrouped')}
          className={`px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all flex items-center space-x-2 border whitespace-nowrap cursor-pointer ${
            selectedGroupId === 'ungrouped'
              ? 'bg-industrial-900 text-white border-industrial-900 shadow-sm'
              : 'bg-white text-industrial-700 border-industrial-200 hover:bg-industrial-50'
          }`}
        >
          <span>Ungrouped</span>
          <span
            className={`px-2 py-0.5 rounded-full text-[10px] ${
              selectedGroupId === 'ungrouped'
                ? 'bg-industrial-700 text-white'
                : 'bg-industrial-100 text-industrial-600'
            }`}
          >
            {ungroupedCount}
          </span>
        </button>

        {groups.map((group) => {
          const gId = group.id || group._id || '';
          const isSelected = selectedGroupId === gId;
          const count = group.model_count ?? 0;

          return (
            <button
              key={gId}
              onClick={() => setSelectedGroupId(gId)}
              className={`px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all flex items-center space-x-2 border whitespace-nowrap cursor-pointer ${
                isSelected
                  ? 'bg-brand-600 text-white border-brand-600 shadow-sm'
                  : 'bg-white text-industrial-700 border-industrial-200 hover:bg-brand-50 hover:border-brand-200'
              }`}
            >
              <Folder className="w-3.5 h-3.5" />
              <span>{group.name}</span>
              <span
                className={`px-2 py-0.5 rounded-full text-[10px] ${
                  isSelected
                    ? 'bg-brand-700 text-white'
                    : 'bg-industrial-100 text-industrial-600'
                }`}
              >
                {count}
              </span>
            </button>
          );
        })}
      </div>

      {models.length === 0 ? (
        <EmptyState
          icon={Cpu}
          title="No Inspection Models Found"
          description="Create your first PatchCore visual inspection model by uploading GOOD reference images."
          actionLabel="Create First Model"
          onAction={() => navigate('/models/create')}
        />
      ) : filteredModels.length === 0 ? (
        <Card className="p-10 text-center space-y-4">
          <Folder className="w-12 h-12 text-industrial-400 mx-auto" />
          <h3 className="text-lg font-bold text-industrial-900">No models in this group</h3>
          <p className="text-sm text-industrial-500 max-w-sm mx-auto">
            There are currently no inspection models assigned to this group filter.
          </p>
          <Button variant="outline" size="sm" onClick={() => setSelectedGroupId('all')}>
            Show All Models
          </Button>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-7">
          {filteredModels.map((model) => {
            const mId = model.id || model._id || '';
            const isActionLoading = actionLoadingId === mId;
            const status = (model.status || 'draft').toLowerCase();

            return (
              <Card
                key={mId}
                className="p-7 flex flex-col justify-between hover:border-brand-300 hover:shadow-md transition-all bg-white border-industrial-200"
              >
                <div className="space-y-5">
                  {/* Header info & Interactive Status Badge */}
                  <div className="flex items-start justify-between">
                    <div className="flex items-center space-x-4">
                      <div className="p-3.5 bg-brand-50 rounded-2xl text-brand-600 border border-brand-200">
                        <Cpu className="w-7 h-7" />
                      </div>
                      <div>
                        <h3 className="font-bold text-xl text-industrial-900 leading-tight">
                          {model.name}
                        </h3>
                        {/* Interactive Group Badge */}
                        <div className="mt-1.5 flex items-center space-x-2">
                          <button
                            onClick={() => setAssignTargetModel(model)}
                            className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-industrial-100 text-industrial-700 hover:bg-brand-50 hover:text-brand-700 border border-industrial-200 hover:border-brand-300 transition-all cursor-pointer"
                            title="Click to change model group"
                          >
                            <Tag className="w-3 h-3 text-brand-600" />
                            <span>{model.group?.name || 'Ungrouped'}</span>
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* Interactive Top-Right Status Badge Button */}
                    <div>
                      {status === 'active' && (
                        <button
                          onClick={() => setDeactivateTarget(model)}
                          disabled={isActionLoading}
                          title="Click to deactivate model"
                          className="px-3 py-1.5 rounded-lg text-xs font-mono font-extrabold tracking-wider uppercase border bg-emerald-100 text-emerald-800 border-emerald-300 hover:bg-emerald-200 hover:border-emerald-400 transition-all flex items-center space-x-1.5 shadow-sm cursor-pointer"
                        >
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          <span>{isActionLoading ? 'SAVING...' : 'ACTIVE'}</span>
                        </button>
                      )}

                      {status === 'inactive' && (
                        <button
                          onClick={() => setActivateTarget(model)}
                          disabled={isActionLoading}
                          title="Click to activate model"
                          className="px-3 py-1.5 rounded-lg text-xs font-mono font-extrabold tracking-wider uppercase border bg-industrial-200 text-industrial-800 border-industrial-300 hover:bg-industrial-300 hover:text-industrial-900 transition-all flex items-center space-x-1.5 shadow-sm cursor-pointer"
                        >
                          <Power className="w-3.5 h-3.5 text-industrial-600" />
                          <span>{isActionLoading ? 'SAVING...' : 'INACTIVE'}</span>
                        </button>
                      )}

                      {status === 'draft' && (
                        <button
                          onClick={() => navigate(`/models/${mId}/build`)}
                          title="Click to configure and build model"
                          className="px-3 py-1.5 rounded-lg text-xs font-mono font-extrabold tracking-wider uppercase border bg-industrial-200 text-industrial-800 border-industrial-300 hover:bg-brand-100 hover:text-brand-800 hover:border-brand-300 transition-all cursor-pointer shadow-sm"
                        >
                          DRAFT
                        </button>
                      )}

                      {status === 'building' && (
                        <div
                          title="Model build in progress..."
                          className="px-3 py-1.5 rounded-lg text-xs font-mono font-extrabold tracking-wider uppercase border bg-amber-100 text-amber-800 border-amber-300 animate-pulse flex items-center space-x-1.5 cursor-not-allowed select-none shadow-sm"
                        >
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          <span>BUILDING</span>
                        </div>
                      )}

                      {status === 'error' && (
                        <button
                          onClick={() => navigate(`/models/${mId}/build`)}
                          title="Click to open configuration and build flow for error recovery"
                          className="px-3 py-1.5 rounded-lg text-xs font-mono font-extrabold tracking-wider uppercase border bg-rose-100 text-rose-800 border-rose-300 hover:bg-rose-200 transition-all flex items-center space-x-1.5 cursor-pointer shadow-sm"
                        >
                          <AlertTriangle className="w-3.5 h-3.5" />
                          <span>ERROR</span>
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Description */}
                  <p className="text-base text-industrial-600 line-clamp-2 min-h-[48px]">
                    {model.description || 'Generic industrial visual inspection model configured with PatchCore engine.'}
                  </p>

                  {/* Specs metadata */}
                  <div className="bg-industrial-50 p-4 rounded-xl border border-industrial-200 space-y-2.5 text-sm font-mono">
                    <div className="flex items-center justify-between text-industrial-600">
                      <span className="flex items-center space-x-2 text-industrial-500">
                        <Images className="w-4 h-4" />
                        <span>Reference Images:</span>
                      </span>
                      <span className="font-bold text-industrial-900">
                        {model.reference_image_count} normal
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-industrial-600">
                      <span className="flex items-center space-x-2 text-industrial-500">
                        <Layers className="w-4 h-4" />
                        <span>Architecture:</span>
                      </span>
                      <span className="font-bold text-brand-700">PatchCore WRN-50</span>
                    </div>
                  </div>
                </div>

                {/* Bottom Card Action Button */}
                <div className="pt-5 mt-5 border-t border-industrial-200 flex items-center justify-between">
                  <span className="text-xs text-industrial-400 font-mono">
                    {model.updated_at ? new Date(model.updated_at).toLocaleDateString() : 'Active'}
                  </span>

                  {status === 'active' && (
                    <div className="flex items-center space-x-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => navigate(`/models/${mId}`)}
                      >
                        View Details
                      </Button>
                      <Button
                        variant="primary"
                        size="sm"
                        onClick={() => navigate(`/inspect?model_id=${mId}`)}
                        icon={<ArrowRight className="w-4 h-4" />}
                      >
                        Inspect
                      </Button>
                    </div>
                  )}

                  {status === 'inactive' && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => navigate(`/models/${mId}`)}
                      icon={<ArrowRight className="w-4 h-4 text-industrial-400" />}
                    >
                      View Details
                    </Button>
                  )}

                  {status === 'draft' && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => navigate(`/models/${mId}/build`)}
                    >
                      Build Model
                    </Button>
                  )}

                  {status === 'building' && (
                    <Button variant="outline" size="sm" disabled>
                      Building...
                    </Button>
                  )}

                  {status === 'error' && (
                    <Button
                      variant="danger"
                      size="sm"
                      onClick={() => setErrorTarget({ model, message: model.error_reason || 'Build or artifact validation failed.' })}
                      icon={<AlertTriangle className="w-4 h-4" />}
                    >
                      View Issue
                    </Button>
                  )}
                </div>
              </Card>
            );
          })}
        </div>
      )}

      {/* Model Group Manager Modal */}
      <ModelGroupManagerModal
        isOpen={isGroupManagerOpen}
        onClose={() => setIsGroupManagerOpen(false)}
        groups={groups}
        onGroupsUpdated={fetchAllData}
      />

      {/* Assign Model Group Modal */}
      <AssignGroupModal
        isOpen={Boolean(assignTargetModel)}
        onClose={() => setAssignTargetModel(null)}
        model={assignTargetModel}
        groups={groups}
        onAssigned={fetchAllData}
      />

      {/* Activation Confirmation Modal */}
      {activateTarget && (
        <div className="fixed inset-0 z-50 bg-industrial-950/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white border border-industrial-200 rounded-2xl p-7 max-w-md w-full shadow-2xl space-y-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3 text-emerald-700 font-bold text-lg">
                <Power className="w-6 h-6 text-emerald-600" />
                <span>Activate model?</span>
              </div>
              <button
                onClick={() => setActivateTarget(null)}
                className="text-industrial-400 hover:text-industrial-600 transition-all"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <p className="text-base text-industrial-700">
              This model will become available for new inspections using its active model version.
            </p>

            <div className="flex items-center justify-end space-x-3 pt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setActivateTarget(null)}
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={handleConfirmActivate}
                disabled={actionLoadingId === (activateTarget.id || activateTarget._id)}
              >
                {actionLoadingId ? 'Activating...' : 'Activate'}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Deactivation Confirmation Modal */}
      {deactivateTarget && (
        <div className="fixed inset-0 z-50 bg-industrial-950/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white border border-industrial-200 rounded-2xl p-7 max-w-md w-full shadow-2xl space-y-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3 text-industrial-900 font-bold text-lg">
                <Power className="w-6 h-6 text-industrial-600" />
                <span>Deactivate model?</span>
              </div>
              <button
                onClick={() => setDeactivateTarget(null)}
                className="text-industrial-400 hover:text-industrial-600 transition-all"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <p className="text-base text-industrial-700">
              This model will no longer be available for new inspections. Its versions, references, and artifacts will be preserved.
            </p>

            <div className="flex items-center justify-end space-x-3 pt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setDeactivateTarget(null)}
              >
                Cancel
              </Button>
              <Button
                variant="danger"
                size="sm"
                onClick={handleConfirmDeactivate}
                disabled={actionLoadingId === (deactivateTarget.id || deactivateTarget._id)}
              >
                {actionLoadingId ? 'Deactivating...' : 'Deactivate'}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Error Reason Modal */}
      {errorTarget && (
        <div className="fixed inset-0 z-50 bg-industrial-950/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white border border-reject-200 rounded-2xl p-7 max-w-lg w-full shadow-2xl space-y-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3 text-reject-600 font-bold text-lg">
                <AlertTriangle className="w-6 h-6" />
                <span>Model Usability Issue</span>
              </div>
              <button
                onClick={() => setErrorTarget(null)}
                className="text-industrial-400 hover:text-industrial-600 transition-all"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-2">
              <h4 className="text-base font-bold text-industrial-900">{errorTarget.model.name}</h4>
              <div className="p-4 bg-reject-50 border border-reject-200 text-reject-800 text-sm font-mono rounded-xl leading-relaxed">
                {errorTarget.message}
              </div>
            </div>

            <div className="flex items-center justify-end space-x-3 pt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setErrorTarget(null)}
              >
                Close
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={() => {
                  const rebuildId = errorTarget.model.id || errorTarget.model._id;
                  setErrorTarget(null);
                  if (rebuildId) {
                    navigate(`/models/${rebuildId}/build`);
                  } else {
                    navigate('/models/create');
                  }
                }}
                icon={<RefreshCw className="w-4 h-4" />}
              >
                Rebuild Model
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
