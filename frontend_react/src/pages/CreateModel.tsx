import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  Upload,
  X,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  Info,
  Images,
} from 'lucide-react';
import {
  getModel,
  getReferenceImages,
  createModel,
  uploadReferenceImages,
  buildModelVersion,
} from '../api/models';
import { getModelGroups } from '../api/model_groups';
import { Model, ModelGroup, ReferenceImage } from '../types';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { LoadingSpinner } from '../components/common/LoadingSpinner';

export const CreateModel: React.FC = () => {
  const navigate = useNavigate();
  const { modelId } = useParams<{ modelId?: string }>();

  const isExistingModel = Boolean(modelId);

  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [name, setName] = useState<string>('');
  const [description, setDescription] = useState<string>('');
  const [selectedGroupId, setSelectedGroupId] = useState<string>('');
  const [availableGroups, setAvailableGroups] = useState<ModelGroup[]>([]);
  const [existingModel, setExistingModel] = useState<Model | null>(null);
  const [existingReferences, setExistingReferences] = useState<ReferenceImage[]>([]);
  const [isLoadingModel, setIsLoadingModel] = useState<boolean>(isExistingModel);
  const [selectedFiles, setSelectedFiles] = useState<{ file: File; preview: string }[]>([]);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [buildStatusMessage, setBuildStatusMessage] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch available model groups for assignment
    getModelGroups()
      .then((groups) => setAvailableGroups(groups))
      .catch(() => setAvailableGroups([]));
  }, []);

  useEffect(() => {
    if (!modelId) return;

    async function loadExistingModel() {
      try {
        setIsLoadingModel(true);
        setError(null);
        const modelData = await getModel(modelId!);
        setExistingModel(modelData);
        setName(modelData.name || '');
        setDescription(modelData.description || '');
        if (modelData.group_id) {
          setSelectedGroupId(modelData.group_id);
        }

        try {
          const refs = await getReferenceImages(modelId!);
          setExistingReferences(refs || []);
        } catch {
          setExistingReferences([]);
        }
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to load existing model details.');
      } finally {
        setIsLoadingModel(false);
      }
    }

    loadExistingModel();
  }, [modelId]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const filesArray = Array.from(e.target.files);
      const newItems = filesArray.map((file) => ({
        file,
        preview: URL.createObjectURL(file),
      }));
      setSelectedFiles((prev) => [...prev, ...newItems]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files) {
      const filesArray = Array.from(e.dataTransfer.files);
      const newItems = filesArray.map((file) => ({
        file,
        preview: URL.createObjectURL(file),
      }));
      setSelectedFiles((prev) => [...prev, ...newItems]);
    }
  };

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => {
      const updated = [...prev];
      URL.revokeObjectURL(updated[index].preview);
      updated.splice(index, 1);
      return updated;
    });
  };

  const handleNextStep = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError('Please provide a name for the model.');
      return;
    }
    setError(null);
    setStep(2);
  };

  const handleCreateAndBuild = async () => {
    const totalRefImages = existingReferences.length + selectedFiles.length;
    if (totalRefImages === 0) {
      setError('Please select or upload at least 1 GOOD reference image to train the normality model.');
      return;
    }

    try {
      setIsSubmitting(true);
      setError(null);
      setStep(3);

      let targetModelId = modelId;

      if (!isExistingModel) {
        // 1. Create Model if brand new
        setBuildStatusMessage('Initializing model registry entry...');
        const created = await createModel(name, description, undefined, selectedGroupId || null);
        targetModelId = created.id || created._id;
      } else {
        setBuildStatusMessage(`Loading existing draft model context (${name})...`);
      }

      if (!targetModelId) {
        throw new Error('Target model ID is missing.');
      }

      // 2. Upload new reference images if added
      if (selectedFiles.length > 0) {
        setBuildStatusMessage(`Uploading ${selectedFiles.length} new GOOD reference images...`);
        const rawFiles = selectedFiles.map((item) => item.file);
        await uploadReferenceImages(targetModelId, rawFiles);
      }

      // 3. Build PatchCore Version for existing/new model
      setBuildStatusMessage('Building PatchCore normality memory bank & calibrating score threshold...');
      await buildModelVersion(targetModelId);

      setBuildStatusMessage('Model build complete! Redirecting to Inspection Workspace...');
      setTimeout(() => {
        navigate(`/inspect?model_id=${targetModelId}`);
      }, 1200);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to build model version.');
      setStep(2);
      setIsSubmitting(false);
    }
  };

  if (isLoadingModel) {
    return <LoadingSpinner label="Fetching existing model details..." size="lg" />;
  }

  const totalRefImages = existingReferences.length + selectedFiles.length;

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6">
      {/* Step Indicator Bar */}
      <div className="bg-white p-4 rounded-xl border border-industrial-200 shadow-sm flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div
            className={`w-7 h-7 rounded-full flex items-center justify-center font-bold text-xs font-mono ${
              step >= 1 ? 'bg-brand-600 text-white' : 'bg-industrial-200 text-industrial-600'
            }`}
          >
            1
          </div>
          <span className={`text-xs font-semibold ${step >= 1 ? 'text-industrial-900' : 'text-industrial-400'}`}>
            {isExistingModel ? 'Build Config' : 'Model Setup'}
          </span>
          <span className="text-industrial-300">/</span>
          <div
            className={`w-7 h-7 rounded-full flex items-center justify-center font-bold text-xs font-mono ${
              step >= 2 ? 'bg-brand-600 text-white' : 'bg-industrial-200 text-industrial-600'
            }`}
          >
            2
          </div>
          <span className={`text-xs font-semibold ${step >= 2 ? 'text-industrial-900' : 'text-industrial-400'}`}>
            GOOD References ({totalRefImages})
          </span>
          <span className="text-industrial-300">/</span>
          <div
            className={`w-7 h-7 rounded-full flex items-center justify-center font-bold text-xs font-mono ${
              step === 3 ? 'bg-brand-600 text-white' : 'bg-industrial-200 text-industrial-600'
            }`}
          >
            3
          </div>
          <span className={`text-xs font-semibold ${step === 3 ? 'text-industrial-900' : 'text-industrial-400'}`}>
            Build Normality Bank
          </span>
        </div>

        {isExistingModel && (
          <span className="text-xs font-mono text-brand-700 bg-brand-50 px-3 py-1 rounded-full border border-brand-200 font-medium">
            Draft Model: {modelId?.slice(-8)}
          </span>
        )}
      </div>

      {error && (
        <div className="p-4 bg-reject-50 border border-reject-200 text-reject-700 text-sm rounded-xl flex items-center space-x-2">
          <AlertCircle className="w-5 h-5 text-reject-600 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Step 1: Model Metadata */}
      {step === 1 && (
        <Card className="p-6">
          <form onSubmit={handleNextStep} className="space-y-5">
            <div>
              <h2 className="text-lg font-bold text-industrial-900">
                {isExistingModel ? `Step 1: Configure Draft Model` : 'Step 1: Create New Model'}
              </h2>
              <p className="text-xs text-industrial-500 mt-1">
                {isExistingModel
                  ? `Pre-populated model parameters loaded from database. Verify or update metadata.`
                  : 'Define the model name, organizational group, and product description.'}
              </p>
            </div>

            <div>
              <label className="block text-xs font-mono font-semibold text-industrial-700 uppercase tracking-wider mb-1.5">
                Model Name <span className="text-reject-600">*</span>
              </label>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. PCB Surface Inspection, Solar Cell Wafer, Tile Coating"
                className="w-full px-3.5 py-2.5 text-base border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none transition-all font-medium text-industrial-900"
              />
            </div>

            {/* Model Group Selection */}
            <div>
              <label className="block text-xs font-mono font-semibold text-industrial-700 uppercase tracking-wider mb-1.5">
                Model Group / Cluster (Optional)
              </label>
              <select
                value={selectedGroupId}
                onChange={(e) => setSelectedGroupId(e.target.value)}
                className="w-full px-3.5 py-2.5 text-base border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none transition-all font-medium text-industrial-900 bg-white"
              >
                <option value="">Ungrouped (Default)</option>
                {availableGroups.map((g) => (
                  <option key={g.id || g._id} value={g.id || g._id}>
                    {g.name}
                  </option>
                ))}
              </select>
              <p className="text-xs text-industrial-500 mt-1">
                Organizes models into categories (e.g., Construction, Manufacturing, Electronics).
              </p>
            </div>

            <div>
              <label className="block text-xs font-mono font-semibold text-industrial-700 uppercase tracking-wider mb-1.5">
                Description (Optional)
              </label>
              <textarea
                rows={3}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe the product, line location, or defect inspection criteria..."
                className="w-full px-3.5 py-2.5 text-base border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none transition-all text-industrial-900"
              />
            </div>

            <div className="pt-4 border-t border-industrial-200 flex justify-end">
              <Button type="submit" variant="primary">
                Next: Upload References →
              </Button>
            </div>
          </form>
        </Card>
      )}

      {/* Step 2: Upload GOOD References */}
      {step === 2 && (
        <Card className="p-6 space-y-6">
          <div>
            <h2 className="text-lg font-bold text-industrial-900">
              {isExistingModel ? `Step 2: GOOD References for ${name}` : 'Step 2: Upload GOOD Reference Images'}
            </h2>
            <div className="mt-2 p-3.5 bg-brand-50 border border-brand-200 rounded-xl text-xs text-brand-800 flex items-start space-x-2.5">
              <Info className="w-4 h-4 text-brand-600 flex-shrink-0 mt-0.5" />
              <div>
                <strong className="font-semibold">Unsupervised Normality Training:</strong> Upload images showing
                defect-free products only. These images define what normal looks like for the PatchCore memory bank.
              </div>
            </div>
          </div>

          {/* Existing References Section */}
          {existingReferences.length > 0 && (
            <div className="p-4 bg-industrial-50 border border-industrial-200 rounded-xl space-y-3">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="flex items-center space-x-2 font-bold text-industrial-800">
                  <Images className="w-4 h-4 text-brand-600" />
                  <span>Existing Stored Reference Images ({existingReferences.length})</span>
                </span>
                <span className="text-emerald-700 bg-emerald-100/80 px-2 py-0.5 rounded font-mono text-xs font-bold border border-emerald-200">
                  Ready in Database
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2 max-h-36 overflow-y-auto">
                {existingReferences.map((ref, idx) => (
                  <div
                    key={ref.id || idx}
                    className="p-2 bg-white rounded-lg border border-industrial-200 text-xs font-mono truncate text-industrial-700 shadow-sm flex items-center space-x-1.5"
                    title={ref.filename}
                  >
                    <CheckCircle2 className="w-3.5 h-3.5 text-pass-600 flex-shrink-0" />
                    <span className="truncate">{ref.filename || `ref_image_${idx + 1}`}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Drag & Drop Area */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="border-2 border-dashed border-industrial-300 hover:border-brand-500 rounded-xl p-8 text-center bg-industrial-50 hover:bg-white transition-all cursor-pointer group"
          >
            <input
              type="file"
              multiple
              accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp"
              onChange={handleFileChange}
              className="hidden"
              id="reference-upload-input"
            />
            <label htmlFor="reference-upload-input" className="cursor-pointer block space-y-2">
              <div className="w-12 h-12 bg-white rounded-full border border-industrial-200 flex items-center justify-center mx-auto text-industrial-500 group-hover:text-brand-600 group-hover:scale-105 transition-all shadow-sm">
                <Upload className="w-6 h-6" />
              </div>
              <p className="text-base font-bold text-industrial-900">
                {existingReferences.length > 0
                  ? 'Click to add MORE GOOD reference images (optional)'
                  : 'Click to upload or drag & drop GOOD reference images'}
              </p>
              <p className="text-xs text-industrial-500 font-mono">PNG, JPG, WEBP, BMP up to 50MB per file</p>
            </label>
          </div>

          {/* Selected New Files Grid */}
          {selectedFiles.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs font-mono text-industrial-600">
                <span>New Files to Upload ({selectedFiles.length})</span>
                <button
                  type="button"
                  onClick={() => setSelectedFiles([])}
                  className="text-reject-600 hover:underline font-semibold"
                >
                  Clear New Files
                </button>
              </div>

              <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-3 max-h-60 overflow-y-auto p-1 border border-industrial-200 rounded-xl bg-white">
                {selectedFiles.map((item, idx) => (
                  <div
                    key={idx}
                    className="relative group rounded-lg border border-industrial-200 overflow-hidden bg-industrial-50 h-20 shadow-sm"
                  >
                    <img src={item.preview} alt="Reference Preview" className="w-full h-full object-cover" />
                    <button
                      type="button"
                      onClick={() => removeFile(idx)}
                      className="absolute top-1 right-1 bg-industrial-900/80 text-white p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity hover:bg-reject-600"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="pt-4 border-t border-industrial-200 flex items-center justify-between">
            <Button variant="outline" onClick={() => setStep(1)}>
              ← Back to Model Setup
            </Button>
            <Button
              variant="primary"
              disabled={totalRefImages === 0}
              onClick={handleCreateAndBuild}
              icon={<Sparkles className="w-4 h-4" />}
            >
              Build PatchCore Model ({totalRefImages} Total Images)
            </Button>
          </div>
        </Card>
      )}

      {/* Step 3: Model Building Progress */}
      {step === 3 && (
        <Card className="p-10 text-center space-y-6">
          <div className="w-16 h-16 bg-brand-50 text-brand-600 rounded-full flex items-center justify-center mx-auto border border-brand-200">
            <Sparkles className="w-8 h-8 animate-spin" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-industrial-900">
              Building PatchCore Model ({name})...
            </h2>
            <p className="text-xs font-mono text-brand-700 mt-2 font-medium bg-brand-50 inline-block px-4 py-1.5 rounded-full border border-brand-200">
              {buildStatusMessage}
            </p>
          </div>

          <div className="max-w-md mx-auto bg-industrial-50 p-4 rounded-xl text-left text-xs font-mono space-y-2 text-industrial-700 border border-industrial-200 shadow-sm">
            <div className="flex items-center space-x-2">
              <CheckCircle2 className="w-4 h-4 text-pass-600 flex-shrink-0" />
              <span>Extracted WideResNet-50 visual feature maps</span>
            </div>
            <div className="flex items-center space-x-2">
              <CheckCircle2 className="w-4 h-4 text-pass-600 flex-shrink-0" />
              <span>Coreset memory bank subsampling (5% ratio)</span>
            </div>
            <div className="flex items-center space-x-2">
              <CheckCircle2 className="w-4 h-4 text-pass-600 flex-shrink-0" />
              <span>Calibrating 95th-percentile anomaly threshold</span>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};
