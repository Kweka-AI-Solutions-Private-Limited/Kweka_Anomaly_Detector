import React, { useState, useEffect } from 'react';
import { Folder, X, Check, AlertTriangle } from 'lucide-react';
import { Model, ModelGroup } from '../../types';
import { updateModel } from '../../api/models';
import { Button } from '../common/Button';

interface AssignGroupModalProps {
  isOpen: boolean;
  onClose: () => void;
  model: Model | null;
  groups: ModelGroup[];
  onAssigned: () => void;
}

export const AssignGroupModal: React.FC<AssignGroupModalProps> = ({
  isOpen,
  onClose,
  model,
  groups,
  onAssigned,
}) => {
  const [selectedGroupId, setSelectedGroupId] = useState<string>('');
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (model) {
      setSelectedGroupId(model.group_id || '');
    }
  }, [model]);

  if (!isOpen || !model) return null;

  const mId = model.id || model._id || '';

  const handleSaveGroupAssignment = async () => {
    try {
      setIsSaving(true);
      setError(null);
      await updateModel(mId, {
        group_id: selectedGroupId || null,
      });
      onAssigned();
      onClose();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update model group.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-industrial-950/60 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-white border border-industrial-200 rounded-2xl p-6 max-w-md w-full shadow-2xl space-y-5">
        <div className="flex items-center justify-between border-b border-industrial-200 pb-3">
          <div className="flex items-center space-x-2.5 text-industrial-900 font-bold text-lg">
            <Folder className="w-5 h-5 text-brand-600" />
            <span>Assign Model Group</span>
          </div>
          <button
            onClick={onClose}
            className="text-industrial-400 hover:text-industrial-600 p-1 rounded-lg transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div>
          <p className="text-xs text-industrial-500">Model:</p>
          <p className="text-base font-bold text-industrial-900 mt-0.5">{model.name}</p>
        </div>

        {error && (
          <div className="p-3 bg-reject-50 border border-reject-200 text-reject-700 text-xs rounded-xl flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-reject-600 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <div className="space-y-2">
          <label className="block text-xs font-mono font-semibold text-industrial-700 uppercase tracking-wider">
            Select Group / Cluster
          </label>
          <select
            value={selectedGroupId}
            onChange={(e) => setSelectedGroupId(e.target.value)}
            className="w-full px-3.5 py-2.5 text-sm border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none text-industrial-900 bg-white font-medium"
          >
            <option value="">Ungrouped (Default)</option>
            {groups.map((g) => {
              const gId = g.id || g._id || '';
              return (
                <option key={gId} value={gId}>
                  {g.name} ({g.model_count ?? 0} models)
                </option>
              );
            })}
          </select>
        </div>

        <div className="flex items-center justify-end space-x-3 pt-2">
          <Button variant="outline" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={handleSaveGroupAssignment}
            disabled={isSaving}
            icon={<Check className="w-4 h-4" />}
          >
            {isSaving ? 'Saving...' : 'Save Assignment'}
          </Button>
        </div>
      </div>
    </div>
  );
};
