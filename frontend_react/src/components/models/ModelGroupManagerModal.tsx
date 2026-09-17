import React, { useState } from 'react';
import { Folder, Plus, Edit2, Trash2, X, AlertTriangle, Check, Layers } from 'lucide-react';
import { ModelGroup } from '../../types';
import { createModelGroup, updateModelGroup, deleteModelGroup } from '../../api/model_groups';
import { Button } from '../common/Button';

interface ModelGroupManagerModalProps {
  isOpen: boolean;
  onClose: () => void;
  groups: ModelGroup[];
  onGroupsUpdated: () => void;
}

export const ModelGroupManagerModal: React.FC<ModelGroupManagerModalProps> = ({
  isOpen,
  onClose,
  groups,
  onGroupsUpdated,
}) => {
  const [newGroupName, setNewGroupName] = useState<string>('');
  const [newGroupDesc, setNewGroupDesc] = useState<string>('');
  const [isCreating, setIsCreating] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Edit mode state
  const [editingGroupId, setEditingGroupId] = useState<string | null>(null);
  const [editName, setEditName] = useState<string>('');
  const [editDesc, setEditDesc] = useState<string>('');
  const [isSavingEdit, setIsSavingEdit] = useState<boolean>(false);

  // Delete confirmation state
  const [deleteTargetGroup, setDeleteTargetGroup] = useState<ModelGroup | null>(null);
  const [isDeleting, setIsDeleting] = useState<boolean>(false);

  if (!isOpen) return null;

  const handleCreateGroup = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newGroupName.trim()) {
      setError('Group name is required.');
      return;
    }
    try {
      setIsCreating(true);
      setError(null);
      await createModelGroup({
        name: newGroupName.trim(),
        description: newGroupDesc.trim() || undefined,
      });
      setNewGroupName('');
      setNewGroupDesc('');
      onGroupsUpdated();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create model group.');
    } finally {
      setIsCreating(false);
    }
  };

  const startEditGroup = (group: ModelGroup) => {
    const gId = group.id || group._id || '';
    setEditingGroupId(gId);
    setEditName(group.name);
    setEditDesc(group.description || '');
    setError(null);
  };

  const handleSaveEdit = async (groupId: string) => {
    if (!editName.trim()) {
      setError('Group name cannot be empty.');
      return;
    }
    try {
      setIsSavingEdit(true);
      setError(null);
      await updateModelGroup(groupId, {
        name: editName.trim(),
        description: editDesc.trim() || undefined,
      });
      setEditingGroupId(null);
      onGroupsUpdated();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update group.');
    } finally {
      setIsSavingEdit(false);
    }
  };

  const handleConfirmDeleteGroup = async () => {
    if (!deleteTargetGroup) return;
    const gId = deleteTargetGroup.id || deleteTargetGroup._id;
    if (!gId) return;

    try {
      setIsDeleting(true);
      setError(null);
      await deleteModelGroup(gId);
      setDeleteTargetGroup(null);
      onGroupsUpdated();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete group.');
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-industrial-950/60 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-white border border-industrial-200 rounded-2xl p-7 max-w-2xl w-full shadow-2xl space-y-6 max-h-[90vh] flex flex-col">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-industrial-200 pb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 bg-brand-50 rounded-xl text-brand-600 border border-brand-200">
              <Folder className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-xl font-black text-industrial-900">Manage Model Groups</h2>
              <p className="text-xs text-industrial-500 mt-0.5">
                Organize visual inspection models into custom domain groups.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-industrial-400 hover:text-industrial-600 p-1 rounded-lg transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {error && (
          <div className="p-3.5 bg-reject-50 border border-reject-200 text-reject-700 text-xs rounded-xl flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-reject-600 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto space-y-6 pr-1">
          {/* Create New Group Form */}
          <form onSubmit={handleCreateGroup} className="bg-industrial-50 p-4 rounded-xl border border-industrial-200 space-y-3">
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-industrial-800 flex items-center space-x-2">
              <Plus className="w-4 h-4 text-brand-600" />
              <span>Create New Group</span>
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <input
                type="text"
                placeholder="Group Name (e.g. Construction)"
                value={newGroupName}
                onChange={(e) => setNewGroupName(e.target.value)}
                className="px-3.5 py-2 text-sm border border-industrial-300 rounded-lg focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none text-industrial-900 bg-white"
              />
              <input
                type="text"
                placeholder="Description (Optional)"
                value={newGroupDesc}
                onChange={(e) => setNewGroupDesc(e.target.value)}
                className="px-3.5 py-2 text-sm border border-industrial-300 rounded-lg focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none text-industrial-900 bg-white"
              />
            </div>
            <div className="flex justify-end pt-1">
              <Button type="submit" variant="primary" size="sm" disabled={isCreating || !newGroupName.trim()}>
                {isCreating ? 'Creating...' : 'Add Group'}
              </Button>
            </div>
          </form>

          {/* Groups List */}
          <div className="space-y-3">
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-industrial-700">
              Existing Groups ({groups.length})
            </h3>

            {groups.length === 0 ? (
              <div className="p-6 text-center text-xs text-industrial-500 border border-dashed border-industrial-200 rounded-xl">
                No model groups created yet. Add your first group above!
              </div>
            ) : (
              <div className="space-y-2.5">
                {groups.map((group) => {
                  const gId = group.id || group._id || '';
                  const isEditing = editingGroupId === gId;

                  if (isEditing) {
                    return (
                      <div key={gId} className="p-4 bg-white border border-brand-300 rounded-xl space-y-3 shadow-sm">
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                          <input
                            type="text"
                            value={editName}
                            onChange={(e) => setEditName(e.target.value)}
                            className="px-3 py-1.5 text-sm border border-industrial-300 rounded-lg font-bold text-industrial-900"
                          />
                          <input
                            type="text"
                            value={editDesc}
                            onChange={(e) => setEditDesc(e.target.value)}
                            placeholder="Description"
                            className="px-3 py-1.5 text-sm border border-industrial-300 rounded-lg text-industrial-700"
                          />
                        </div>
                        <div className="flex justify-end space-x-2">
                          <Button variant="outline" size="sm" onClick={() => setEditingGroupId(null)}>
                            Cancel
                          </Button>
                          <Button
                            variant="primary"
                            size="sm"
                            onClick={() => handleSaveEdit(gId)}
                            disabled={isSavingEdit}
                            icon={<Check className="w-3.5 h-3.5" />}
                          >
                            Save
                          </Button>
                        </div>
                      </div>
                    );
                  }

                  return (
                    <div
                      key={gId}
                      className="p-4 bg-white border border-industrial-200 rounded-xl flex items-center justify-between hover:border-industrial-300 transition-all shadow-sm"
                    >
                      <div className="space-y-1">
                        <div className="flex items-center space-x-3">
                          <span className="font-bold text-base text-industrial-900">{group.name}</span>
                          <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-industrial-100 text-industrial-700 border border-industrial-200">
                            {group.model_count ?? 0} models
                          </span>
                        </div>
                        {group.description && (
                          <p className="text-xs text-industrial-600 line-clamp-1">{group.description}</p>
                        )}
                      </div>

                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => startEditGroup(group)}
                          className="p-1.5 text-industrial-500 hover:text-brand-600 hover:bg-industrial-100 rounded-lg transition-all"
                          title="Edit Group"
                        >
                          <Edit2 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setDeleteTargetGroup(group)}
                          className="p-1.5 text-industrial-500 hover:text-reject-600 hover:bg-reject-50 rounded-lg transition-all"
                          title="Delete Group"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="border-t border-industrial-200 pt-4 flex justify-end">
          <Button variant="outline" size="sm" onClick={onClose}>
            Done
          </Button>
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      {deleteTargetGroup && (
        <div className="fixed inset-0 z-60 bg-industrial-950/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white border border-reject-200 rounded-2xl p-6 max-w-md w-full shadow-2xl space-y-4">
            <div className="flex items-center space-x-3 text-reject-600 font-bold text-lg">
              <AlertTriangle className="w-6 h-6" />
              <span>Delete Group "{deleteTargetGroup.name}"?</span>
            </div>

            <p className="text-xs text-industrial-700 leading-relaxed bg-industrial-50 p-3 rounded-xl border border-industrial-200">
              <strong>Safe Unassign Warning:</strong> Deleting this group will move its{' '}
              <strong>{deleteTargetGroup.model_count ?? 0} member model(s)</strong> to <em>Ungrouped</em> status.
              <br /><br />
              No models, PatchCore memory banks, model versions, or historical inspection data will be deleted.
            </p>

            <div className="flex items-center justify-end space-x-3 pt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setDeleteTargetGroup(null)}
              >
                Cancel
              </Button>
              <Button
                variant="danger"
                size="sm"
                onClick={handleConfirmDeleteGroup}
                disabled={isDeleting}
              >
                {isDeleting ? 'Deleting...' : 'Confirm Delete Group'}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
