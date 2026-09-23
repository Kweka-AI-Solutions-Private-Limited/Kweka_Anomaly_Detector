import React, { useState, useRef, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Leaf,
  Upload,
  X,
  AlertCircle,
  CheckCircle2,
  RefreshCw,
  FileCheck,
  Sprout,
  Cpu,
  Info,
  ExternalLink,
  Sparkles,
  FlaskConical,
  ShieldCheck,
  Building2,
  History,
  Trash2,
  Clock,
  ChevronRight,
  MessageSquare,
  ThumbsUp,
  ThumbsDown,
  Send,
  Search,
  BookOpen,
  Filter,
  Check,
} from 'lucide-react';
import { Card } from '../components/common/Card';
import {
  analyzeLeafDisease,
  getLeafDiseaseHistory,
  getLeafDiseaseRun,
  deleteLeafDiseaseRun,
  clearLeafDiseaseHistory,
  getNaclCatalog,
  submitLeafDiseaseFeedback,
  LeafDiseaseAnalysisResponse,
} from '../api/leafDisease';

const CROPS = [
  { value: '', label: 'Auto-detect Crop (Recommended)' },
  { value: 'Cucurbits', label: 'Cucurbits (Cucumber / Melon / Squash / Pumpkin)' },
  { value: 'Pear', label: 'Pear (Pyrus)' },
  { value: 'Apple', label: 'Apple' },
  { value: 'Grape', label: 'Grape' },
  { value: 'Tomato', label: 'Tomato' },
  { value: 'Rice / Paddy', label: 'Rice / Paddy' },
  { value: 'Cotton', label: 'Cotton' },
  { value: 'Chilli / Pepper', label: 'Chilli / Pepper' },
  { value: 'Maize / Corn', label: 'Maize / Corn' },
  { value: 'Potato', label: 'Potato' },
  { value: 'Wheat', label: 'Wheat' },
  { value: 'Sugarcane', label: 'Sugarcane' },
  { value: 'Groundnut / Peanut', label: 'Groundnut / Peanut' },
];

export const LeafDiseasePage: React.FC = () => {
  const { runId } = useParams<{ runId?: string }>();

  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [selectedCrop, setSelectedCrop] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [currentStage, setCurrentStage] = useState<string>('');
  const [analysisResult, setAnalysisResult] = useState<LeafDiseaseAnalysisResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // History State
  const [historyList, setHistoryList] = useState<LeafDiseaseAnalysisResponse[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState<boolean>(false);
  const [showHistoryModal, setShowHistoryModal] = useState<boolean>(false);
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);

  // NACL Catalog Modal State
  const [showCatalogModal, setShowCatalogModal] = useState<boolean>(false);
  const [catalogData, setCatalogData] = useState<{ total_count: number; categories: Record<string, any[]> } | null>(null);
  const [isLoadingCatalog, setIsLoadingCatalog] = useState<boolean>(false);
  const [catalogActiveCategory, setCatalogActiveCategory] = useState<string>('All');
  const [catalogSearch, setCatalogSearch] = useState<string>('');

  // Feedback State
  const [feedbackRating, setFeedbackRating] = useState<'UP' | 'DOWN' | null>(null);
  const [feedbackCategory, setFeedbackCategory] = useState<string>('DIAGNOSIS_CORRECT');
  const [feedbackComments, setFeedbackComments] = useState<string>('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<boolean>(false);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState<boolean>(false);

  // Suggested Crop Confirmation State
  const [suggestedCropChoice, setSuggestedCropChoice] = useState<string>('');

  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchHistory = async () => {
    setIsLoadingHistory(true);
    try {
      const data = await getLeafDiseaseHistory(50);
      setHistoryList(data);
    } catch (err) {
      console.warn('Failed to fetch leaf disease run history:', err);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const loadSpecificRun = async (id: string) => {
    try {
      const runData = await getLeafDiseaseRun(id);
      setAnalysisResult(runData);
      setActiveHistoryId(id);
    } catch (err) {
      console.warn(`Failed to load run '${id}':`, err);
    }
  };

  const fetchCatalog = async () => {
    if (catalogData) return;
    setIsLoadingCatalog(true);
    try {
      const data = await getNaclCatalog();
      setCatalogData(data);
    } catch (err) {
      console.warn('Failed to fetch NACL catalog:', err);
    } finally {
      setIsLoadingCatalog(false);
    }
  };

  useEffect(() => {
    fetchHistory();
    if (runId) {
      loadSpecificRun(runId);
    }
  }, [runId]);

  const handleOpenCatalog = () => {
    setShowCatalogModal(true);
    fetchCatalog();
  };

  const handleSendFeedback = async () => {
    setIsSubmittingFeedback(true);
    try {
      await submitLeafDiseaseFeedback({
        analysis_id: analysisResult?.analysis_id,
        rating: feedbackRating || undefined,
        feedback_category: feedbackCategory,
        comments: feedbackComments,
        crop_name: analysisResult?.crop?.crop_name,
        disease_name: analysisResult?.disease?.disease_name,
      });
      setFeedbackSubmitted(true);
    } catch (err) {
      console.error('Failed to submit feedback:', err);
    } finally {
      setIsSubmittingFeedback(false);
    }
  };

  const handleSelectHistoryRun = (run: LeafDiseaseAnalysisResponse) => {
    // Open historical analysis run in a new browser tab
    window.open(`/leaf-disease/${run.analysis_id}`, '_blank');
  };

  const handleDeleteRun = async (e: React.MouseEvent, analysisId: string) => {
    e.stopPropagation();
    try {
      await deleteLeafDiseaseRun(analysisId);
      setHistoryList((prev) => prev.filter((r) => r.analysis_id !== analysisId));
      if (analysisResult?.analysis_id === analysisId) {
        setAnalysisResult(null);
        setActiveHistoryId(null);
      }
    } catch (err) {
      console.error('Failed to delete leaf disease run:', err);
    }
  };

  const handleClearHistory = async () => {
    if (!window.confirm('Are you sure you want to clear all leaf disease analysis run history?')) return;
    try {
      await clearLeafDiseaseHistory();
      setHistoryList([]);
      setAnalysisResult(null);
      setActiveHistoryId(null);
    } catch (err) {
      console.error('Failed to clear leaf disease history:', err);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    addFiles(Array.from(e.target.files));
  };

  const addFiles = (newFiles: File[]) => {
    setErrorMsg(null);

    const validNewFiles: File[] = [];
    const newPreviews: string[] = [];

    const totalAllowed = 5 - selectedFiles.length;
    if (totalAllowed <= 0) {
      setErrorMsg('Maximum 5 images allowed per analysis.');
      return;
    }

    const filesToProcess = newFiles.slice(0, totalAllowed);

    filesToProcess.forEach((file) => {
      if (!file.type.startsWith('image/')) {
        setErrorMsg(`File '${file.name}' is not a valid image format.`);
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        setErrorMsg(`File '${file.name}' exceeds the 10MB limit.`);
        return;
      }
      validNewFiles.push(file);
      newPreviews.push(URL.createObjectURL(file));
    });

    setSelectedFiles((prev) => [...prev, ...validNewFiles]);
    setPreviewUrls((prev) => [...prev, ...newPreviews]);
  };

  const handleRemoveImage = (index: number) => {
    URL.revokeObjectURL(previewUrls[index]);
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    setPreviewUrls((prev) => prev.filter((_, i) => i !== index));
  };

  const handleClearAll = () => {
    previewUrls.forEach((url) => URL.revokeObjectURL(url));
    setSelectedFiles([]);
    setPreviewUrls([]);
    setAnalysisResult(null);
    setErrorMsg(null);
    setActiveHistoryId(null);
  };

  const handleAnalyze = async () => {
    if (selectedFiles.length === 0) {
      setErrorMsg('Please select at least one leaf image to analyze.');
      return;
    }

    setIsAnalyzing(true);
    setErrorMsg(null);
    setAnalysisResult(null);

    try {
      setCurrentStage('Validating image quality & format...');
      await new Promise((r) => setTimeout(r, 200));

      setCurrentStage('Identifying crop & plant species (Pl@ntNet / Gemini)...');
      await new Promise((r) => setTimeout(r, 200));

      setCurrentStage('Executing primary pathology detection...');
      const res = await analyzeLeafDisease(selectedFiles, selectedCrop);

      setCurrentStage('Retrieving NACL RAG product recommendations & Gemini advisory...');
      await new Promise((r) => setTimeout(r, 200));

      setAnalysisResult(res);
      setActiveHistoryId(res.analysis_id);
      fetchHistory();
    } catch (err: any) {
      console.error('[ERROR] Leaf disease analysis failed:', err);
      setErrorMsg(
        err.response?.data?.detail ||
        'Failed to complete leaf disease analysis. Please ensure backend server is active.'
      );
    } finally {
      setIsAnalyzing(false);
      setCurrentStage('');
    }
  };

  const naclRecs = analysisResult?.nacl_recommendations;

  return (
    <div className="w-full px-4 lg:px-8 py-6 space-y-8 select-none">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-emerald-950 via-industrial-950 to-industrial-900 border border-emerald-800/40 text-white p-6 rounded-2xl shadow-xl space-y-3">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center space-x-3.5">
            <div className="p-3 rounded-2xl bg-emerald-500/20 border border-emerald-500/30 text-emerald-400">
              <Leaf className="w-7 h-7" />
            </div>
            <div>
              <div className="flex items-center space-x-2.5">
                <span className="font-extrabold text-2xl text-white tracking-tight">
                  Leaf Pathology & Agrochemical Advisory
                </span>
                <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-xs font-mono font-bold uppercase tracking-wider">
                  NACL Industries AI • RAG & Gemini Vision
                </span>
              </div>
              <p className="text-xs text-industrial-400 font-mono mt-1">
                Image Quality → Crop Identification → Pathogen Diagnosis → NACL Product Recommendation Engine
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <button
              type="button"
              onClick={() => setShowHistoryModal(true)}
              className="flex items-center space-x-2 text-xs font-mono bg-emerald-900/80 hover:bg-emerald-800 px-3.5 py-2 rounded-xl border border-emerald-600/50 text-emerald-200 transition-colors shadow-sm"
            >
              <History className="w-4 h-4 text-emerald-400" />
              <span>Run History ({historyList.length})</span>
            </button>
            <button
              type="button"
              onClick={handleOpenCatalog}
              className="flex items-center space-x-2 text-xs font-mono bg-industrial-800/90 hover:bg-industrial-800 px-3.5 py-2 rounded-xl border border-emerald-500/50 text-emerald-300 transition-all hover:border-emerald-400 shadow-sm cursor-pointer"
            >
              <Building2 className="w-4.5 h-4.5 text-emerald-400" />
              <span>NACL Catalog ({catalogData?.total_count || 59} Products)</span>
              <BookOpen className="w-3.5 h-3.5 text-emerald-400 ml-1" />
            </button>
          </div>
        </div>
      </div>

      {/* Top Section: Upload Box & Pathology Summary Dashboard */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Image Upload & Controls (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
          <Card className="p-6 space-y-5 bg-white border-industrial-200 shadow-sm">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-extrabold text-industrial-900 flex items-center space-x-2">
                <Upload className="w-4.5 h-4.5 text-emerald-600" />
                <span>Upload Leaf Images</span>
              </h2>
              <span className="text-xs font-mono font-bold text-industrial-500">
                {selectedFiles.length} / 5 Images
              </span>
            </div>

            {/* Drop Zone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (e.dataTransfer.files) {
                  addFiles(Array.from(e.dataTransfer.files));
                }
              }}
              className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all ${selectedFiles.length >= 5
                  ? 'border-industrial-300 bg-industrial-50 opacity-60 cursor-not-allowed'
                  : 'border-industrial-300 hover:border-emerald-500 hover:bg-emerald-50/30 bg-industrial-50/50'
                }`}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/jpeg,image/png,image/webp"
                multiple
                className="hidden"
                disabled={selectedFiles.length >= 5}
              />
              <div className="flex flex-col items-center space-y-2">
                <div className="p-3 rounded-full bg-emerald-100 text-emerald-600">
                  <Upload className="w-6 h-6" />
                </div>
                <p className="text-sm font-bold text-industrial-800">
                  Click or Drag & Drop Leaf Images
                </p>
                <p className="text-xs text-industrial-500 font-mono">
                  JPEG, PNG, WEBP • Max 10MB per file
                </p>
              </div>
            </div>

            {/* Previews */}
            {previewUrls.length > 0 && (
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-industrial-700 uppercase tracking-wider font-mono">
                    Selected Previews
                  </span>
                  <button
                    type="button"
                    onClick={handleClearAll}
                    className="text-xs text-reject-600 hover:text-reject-700 font-semibold"
                  >
                    Clear All
                  </button>
                </div>
                <div className="grid grid-cols-4 gap-2">
                  {previewUrls.map((url, idx) => (
                    <div
                      key={idx}
                      className="relative group rounded-lg overflow-hidden border border-industrial-200 aspect-square bg-industrial-900"
                    >
                      <img
                        src={url}
                        alt={`Leaf preview ${idx + 1}`}
                        className="w-full h-full object-cover"
                      />
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemoveImage(idx);
                        }}
                        className="absolute top-1 right-1 bg-industrial-900/80 hover:bg-reject-600 text-white p-1 rounded-full opacity-90 transition-colors"
                        title="Remove image"
                      >
                        <X className="w-3 h-3" />
                      </button>
                      <span className="absolute bottom-1 left-1 bg-industrial-950/80 text-white font-mono text-[9px] px-1 py-0.5 rounded">
                        #{idx + 1}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Crop Selector */}
            <div className="space-y-2 pt-2 border-t border-industrial-100">
              <label className="block text-xs font-extrabold text-industrial-800 uppercase tracking-wider font-mono">
                Plant / Crop Species (Optional)
              </label>
              <select
                value={selectedCrop}
                onChange={(e) => setSelectedCrop(e.target.value)}
                className="w-full px-3.5 py-2.5 rounded-xl border border-industrial-300 text-sm font-medium text-industrial-900 bg-white focus:ring-2 focus:ring-emerald-500 focus:outline-none"
              >
                {CROPS.map((c) => (
                  <option key={c.value} value={c.value}>
                    {c.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Error Message */}
            {errorMsg && (
              <div className="p-3 rounded-xl bg-reject-50 border border-reject-200 text-reject-700 text-xs flex items-start space-x-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5 text-reject-600" />
                <span className="font-medium leading-relaxed">{errorMsg}</span>
              </div>
            )}

            {/* Action Button */}
            <button
              type="button"
              onClick={handleAnalyze}
              disabled={isAnalyzing || selectedFiles.length === 0}
              className={`w-full py-3.5 px-4 rounded-xl text-sm font-extrabold text-white flex items-center justify-center space-x-2 shadow-sm transition-all ${isAnalyzing || selectedFiles.length === 0
                  ? 'bg-industrial-400 cursor-not-allowed opacity-60'
                  : 'bg-emerald-600 hover:bg-emerald-700 active:scale-[0.99]'
                }`}
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Executing Pipeline...</span>
                </>
              ) : (
                <>
                  <Leaf className="w-4.5 h-4.5" />
                  <span>Diagnose & Recommend Treatment</span>
                </>
              )}
            </button>

            {/* Progress Tracker */}
            {isAnalyzing && (
              <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 space-y-1.5 animate-pulse">
                <div className="flex items-center space-x-2 text-xs font-bold text-emerald-800">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-emerald-600" />
                  <span>{currentStage}</span>
                </div>
              </div>
            )}
          </Card>

          {/* AI Agronomist Feedback & Model Tuning Card (Visible ONLY after executing an analysis run) */}
          {analysisResult && (
            <Card className="p-5 bg-white border-industrial-200 space-y-4 shadow-sm animate-in fade-in">
              <div className="flex items-center justify-between border-b border-industrial-100 pb-2.5">
                <div className="flex items-center space-x-2">
                  <MessageSquare className="w-4.5 h-4.5 text-emerald-600" />
                  <h3 className="text-xs font-extrabold text-industrial-900 uppercase font-mono tracking-wider">
                    AI Agronomist Feedback
                  </h3>
                </div>
                <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 text-[10px] font-mono font-bold uppercase">
                  Model Tuning
                </span>
              </div>

              <p className="text-xs text-industrial-600 font-medium leading-relaxed">
                Provide feedback on this diagnosis to help tune Gemini AI pathology algorithms and improve crop recognition.
              </p>

              {feedbackSubmitted ? (
                <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-center space-y-2">
                  <CheckCircle2 className="w-6 h-6 text-emerald-600 mx-auto" />
                  <p className="text-xs font-bold text-emerald-900">Feedback Submitted!</p>
                  <p className="text-[11px] text-emerald-700 font-mono">
                    Logged into dataset tuning pipeline for Gemini plant vision models.
                  </p>
                  <button
                    type="button"
                    onClick={() => {
                      setFeedbackSubmitted(false);
                      setFeedbackRating(null);
                      setFeedbackComments('');
                    }}
                    className="text-[11px] font-bold text-emerald-800 underline hover:text-emerald-950 pt-1"
                  >
                    Submit Additional Feedback
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  {/* Thumbs Up / Down */}
                  <div className="flex items-center justify-center space-x-3">
                    <button
                      type="button"
                      onClick={() => setFeedbackRating('UP')}
                      className={`flex-1 py-2 px-3 rounded-xl border flex items-center justify-center space-x-2 text-xs font-mono font-bold transition-all ${feedbackRating === 'UP'
                          ? 'bg-emerald-600 text-white border-emerald-700 shadow-sm'
                          : 'bg-industrial-50 border-industrial-200 text-industrial-700 hover:bg-emerald-50 hover:border-emerald-300'
                        }`}
                    >
                      <ThumbsUp className="w-4 h-4" />
                      <span>Accurate AI</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setFeedbackRating('DOWN')}
                      className={`flex-1 py-2 px-3 rounded-xl border flex items-center justify-center space-x-2 text-xs font-mono font-bold transition-all ${feedbackRating === 'DOWN'
                          ? 'bg-reject-600 text-white border-reject-700 shadow-sm'
                          : 'bg-industrial-50 border-industrial-200 text-industrial-700 hover:bg-reject-50 hover:border-reject-300'
                        }`}
                    >
                      <ThumbsDown className="w-4 h-4" />
                      <span>Needs Tuning</span>
                    </button>
                  </div>

                  {/* Category Select */}
                  <div className="space-y-1">
                    <label className="block text-[10px] font-mono font-extrabold text-industrial-600 uppercase">
                      Feedback Category
                    </label>
                    <select
                      value={feedbackCategory}
                      onChange={(e) => setFeedbackCategory(e.target.value)}
                      className="w-full px-3 py-2 rounded-xl border border-industrial-300 text-xs font-medium text-industrial-900 bg-white focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                    >
                      <option value="DIAGNOSIS_CORRECT">Diagnosis & Crop Correct</option>
                      <option value="INCORRECT_CROP">Incorrect Plant Species Identified</option>
                      <option value="INCORRECT_PATHOLOGY">Incorrect Pathology / Disease</option>
                      <option value="TREATMENT_HELPFUL">NACL Recommendations Helpful</option>
                      <option value="OTHER">General Feedback / Edge Case</option>
                    </select>
                  </div>

                  {/* Comments */}
                  <div className="space-y-1">
                    <label className="block text-[10px] font-mono font-extrabold text-industrial-600 uppercase">
                      Agronomist Notes / Actual Crop
                    </label>
                    <textarea
                      rows={2}
                      value={feedbackComments}
                      onChange={(e) => setFeedbackComments(e.target.value)}
                      placeholder="E.g. Actual plant is Cucumber, symptoms match Powdery Mildew..."
                      className="w-full px-3 py-2 rounded-xl border border-industrial-300 text-xs text-industrial-900 focus:ring-2 focus:ring-emerald-500 focus:outline-none resize-none font-sans"
                    />
                  </div>

                  {/* Submit Button */}
                  <button
                    type="button"
                    onClick={handleSendFeedback}
                    disabled={isSubmittingFeedback}
                    className="w-full py-2.5 px-3 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-extrabold flex items-center justify-center space-x-1.5 transition-all shadow-sm"
                  >
                    {isSubmittingFeedback ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <>
                        <Send className="w-3.5 h-3.5" />
                        <span>Submit Agronomist Feedback</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </Card>
          )}
        </div>

        {/* Right Column: Diagnostic Result Dashboard (8 cols) */}
        <div className="lg:col-span-8">
          {!analysisResult && !isAnalyzing && (
            <Card className="p-12 text-center bg-white border-industrial-200 space-y-4 shadow-sm h-full flex flex-col items-center justify-center">
              <div className="w-14 h-14 rounded-2xl bg-emerald-100 text-emerald-600 flex items-center justify-center mx-auto">
                <Sprout className="w-7 h-7" />
              </div>
              <div className="space-y-1 max-w-lg">
                <h3 className="text-lg font-extrabold text-industrial-900">
                  Ready for Leaf Analysis & NACL Treatment Advisory
                </h3>
                <p className="text-xs text-industrial-500 font-medium leading-relaxed">
                  Upload leaf photos showing a single disease on a single plant to identify the species, diagnose pathology (e.g., European Pear Rust), and retrieve official NACL product recommendations with Gemini AI advisories.
                </p>
              </div>
            </Card>
          )}

          {analysisResult && (
            <div className="space-y-4">
              {/* Status Header */}
              <Card className="p-4 bg-white border-industrial-200 shadow-sm flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center space-x-3">
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-mono font-bold tracking-wider uppercase ${analysisResult.status === 'SUCCESS'
                        ? 'bg-pass-100 text-pass-800 border border-pass-300'
                        : analysisResult.status === 'UNCERTAIN'
                          ? 'bg-amber-100 text-amber-800 border border-amber-300'
                          : 'bg-reject-100 text-reject-800 border border-reject-300'
                      }`}
                  >
                    STATUS: {analysisResult.status}
                  </span>
                  <span className="text-xs text-industrial-500 font-mono font-bold">
                    ID: {analysisResult.analysis_id}
                  </span>
                </div>
                <span className="text-xs text-industrial-400 font-mono">
                  {new Date(analysisResult.timestamp).toLocaleString()}
                </span>
              </Card>

              {/* Crop & Disease Result Cards Side-by-Side */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Crop Card */}
                <Card className="p-5 bg-white border-industrial-200 space-y-3 shadow-sm">
                  <div className="flex items-center justify-between border-b border-industrial-100 pb-2">
                    <span className="text-xs font-extrabold text-industrial-500 uppercase tracking-wider font-mono flex items-center space-x-1.5">
                      <Sprout className="w-4 h-4 text-emerald-600" />
                      <span>Identified Crop Species</span>
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${analysisResult.crop.status === 'CONFIRMED'
                          ? 'bg-emerald-100 text-emerald-800'
                          : 'bg-amber-100 text-amber-800'
                        }`}
                    >
                      {analysisResult.crop.status}
                    </span>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xl font-extrabold text-industrial-900">
                      {analysisResult.crop.crop_name}
                    </p>
                    <p className="text-xs text-industrial-500 font-mono">
                      Source: {analysisResult.crop.source}
                    </p>
                  </div>
                  <p className="text-xs text-industrial-600 leading-relaxed font-medium bg-industrial-50 p-2.5 rounded-lg border border-industrial-100">
                    {analysisResult.crop.evidence_note}
                  </p>

                  {/* Interactive Crop Species Selection Dropdown (Prompted when Gemini confidence is uncertain) */}
                  {(analysisResult.crop.status === 'UNCERTAIN' ||
                    analysisResult.diagnosis?.is_uncertain ||
                    analysisResult.crop.crop_name === 'Unknown') && (
                      <div className="mt-3 p-3.5 rounded-xl bg-amber-50 border border-amber-300 space-y-2.5">
                        <div className="flex items-start space-x-2 text-amber-900">
                          <AlertCircle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                          <div className="space-y-1 text-xs">
                            <p className="font-extrabold uppercase font-mono tracking-wide text-amber-950">
                              Suggested Action: Select Plant Species
                            </p>
                            <p className="text-amber-800 leading-normal font-medium">
                              Auto-detect confidence is low. Please select the exact crop species below to confirm diagnosis and unlock verified NACL agrochemical recommendations:
                            </p>
                          </div>
                        </div>

                        <div className="flex items-center space-x-2">
                          <select
                            value={selectedCrop}
                            onChange={(e) => setSelectedCrop(e.target.value)}
                            className="flex-1 px-3 py-2 rounded-lg border border-amber-400 text-xs font-bold text-industrial-900 bg-white focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                          >
                            <option value="">-- Select Plant Species --</option>
                            {CROPS.filter((c) => c.value !== '').map((c) => (
                              <option key={c.value} value={c.value}>
                                {c.label}
                              </option>
                            ))}
                          </select>
                          <button
                            type="button"
                            onClick={() => handleAnalyze()}
                            disabled={!selectedCrop || isAnalyzing}
                            className="px-3.5 py-2 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-bold font-mono transition-all disabled:opacity-50 shrink-0 flex items-center space-x-1"
                          >
                            {isAnalyzing ? (
                              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                            ) : (
                              <>
                                <Check className="w-3.5 h-3.5" />
                                <span>Confirm Crop</span>
                              </>
                            )}
                          </button>
                        </div>
                      </div>
                    )}
                </Card>

                {/* Disease Card */}
                <Card className="p-5 bg-white border-industrial-200 space-y-3 shadow-sm">
                  <div className="flex items-center justify-between border-b border-industrial-100 pb-2">
                    <span className="text-xs font-extrabold text-industrial-500 uppercase tracking-wider font-mono flex items-center space-x-1.5">
                      <Cpu className="w-4 h-4 text-emerald-600" />
                      <span>Diagnosed Pathology</span>
                    </span>
                    {analysisResult.disease?.is_mock && (
                      <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-800 text-[10px] font-mono font-bold uppercase">
                        MOCK PROVIDER
                      </span>
                    )}
                  </div>
                  <div className="space-y-1">
                    <p className="text-xl font-extrabold text-industrial-900">
                      {analysisResult.disease?.disease_name || 'No Pathology Detected'}
                    </p>
                    <div className="flex items-center justify-between text-xs text-industrial-500 font-mono">
                      <span>Provider: {analysisResult.disease?.provider_name}</span>
                      {analysisResult.disease?.confidence && (
                        <span className="font-bold text-industrial-800">
                          Conf: {(analysisResult.disease.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </div>
                  {analysisResult.disease?.candidates && analysisResult.disease.candidates.length > 0 && (
                    <div className="space-y-1 pt-1 border-t border-industrial-100">
                      <p className="text-[10px] font-mono font-bold text-industrial-500 uppercase">
                        Candidates:
                      </p>
                      {analysisResult.disease.candidates.map((c, i) => (
                        <div key={i} className="flex justify-between text-xs font-mono text-industrial-600">
                          <span>• {c.disease_name}</span>
                          <span>{c.confidence ? `${(c.confidence * 100).toFixed(0)}%` : ''}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </Card>
              </div>

              {/* Diagnosis Evidence Validation */}
              {analysisResult.diagnosis && (
                <Card className="p-4 bg-white border-industrial-200 space-y-2.5 shadow-sm">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <FileCheck className="w-4 h-4 text-emerald-600" />
                      <h3 className="text-xs font-extrabold text-industrial-900 font-mono uppercase tracking-wider">
                        Diagnosis Evidence Validation
                      </h3>
                    </div>
                    <span
                      className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase ${analysisResult.diagnosis.evidence_level === 'HIGH'
                          ? 'bg-pass-100 text-pass-800 border border-pass-300'
                          : analysisResult.diagnosis.evidence_level === 'MEDIUM'
                            ? 'bg-amber-100 text-amber-800 border border-amber-300'
                            : 'bg-reject-100 text-reject-800 border border-reject-300'
                        }`}
                    >
                      EVIDENCE: {analysisResult.diagnosis.evidence_level}
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-center font-mono text-xs">
                    <div className="p-2 rounded-xl bg-industrial-50 border border-industrial-200">
                      <span className="text-industrial-400 font-bold block text-[9px]">IMAGE QUALITY</span>
                      <span className="font-extrabold text-industrial-900 text-xs">{analysisResult.diagnosis.image_quality_status}</span>
                    </div>
                    <div className="p-2 rounded-xl bg-industrial-50 border border-industrial-200">
                      <span className="text-industrial-400 font-bold block text-[9px]">CROP COMPATIBILITY</span>
                      <span className="font-extrabold text-industrial-900 text-xs">{analysisResult.diagnosis.crop_compatibility_status}</span>
                    </div>
                    <div className="p-2 rounded-xl bg-industrial-50 border border-industrial-200">
                      <span className="text-industrial-400 font-bold block text-[9px]">CONSISTENCY</span>
                      <span className="font-extrabold text-industrial-900 text-xs">{analysisResult.diagnosis.consistency_status}</span>
                    </div>
                  </div>
                </Card>
              )}

              {/* MAIN TREATMENT & NACL AGROCHEMICAL RECOMMENDATIONS SECTION */}
              {naclRecs && (
                <div className="space-y-4 pt-2 border-t border-industrial-200 animate-in fade-in">
                  {/* Section Header */}
                  <div className="flex items-center justify-between flex-wrap gap-4">
                    <div className="flex items-center space-x-3">
                      <div className="p-2.5 rounded-2xl bg-emerald-100 text-emerald-800 border border-emerald-200">
                        <FlaskConical className="w-5 h-5" />
                      </div>
                      <div>
                        <h3 className="text-lg font-extrabold text-industrial-900 tracking-tight">
                          NACL Agrochemical Treatment Advisory
                        </h3>
                        <p className="text-xs text-industrial-500 font-mono">
                          RAG Database Match • Active Ingredient Chemistry • Gemini AI Advisory
                        </p>
                      </div>
                    </div>
                    <span className="px-3.5 py-1.5 rounded-full bg-emerald-100 text-emerald-800 text-xs font-mono font-extrabold border border-emerald-300">
                      {naclRecs.recommendations.length} Recommended Products
                    </span>
                  </div>

                  {/* Gemini AI Agronomist Advisory Card */}
                  <Card
                    className={`p-5 border text-white rounded-2xl shadow-xl space-y-3.5 ${analysisResult?.diagnosis?.crop_compatibility_status === 'MISMATCHED_HOST' || naclRecs.safety_disclaimer?.includes('CAUTION')
                        ? 'bg-gradient-to-br from-amber-950 via-industrial-900 to-industrial-950 border-amber-600/50'
                        : 'bg-gradient-to-br from-emerald-950 via-industrial-900 to-industrial-950 border-emerald-600/40'
                      }`}
                  >
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <div className="flex items-center space-x-2.5">
                        <div
                          className={`p-2 rounded-xl border ${analysisResult?.diagnosis?.crop_compatibility_status === 'MISMATCHED_HOST' || naclRecs.safety_disclaimer?.includes('CAUTION')
                              ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
                              : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
                            }`}
                        >
                          {analysisResult?.diagnosis?.crop_compatibility_status === 'MISMATCHED_HOST' || naclRecs.safety_disclaimer?.includes('CAUTION') ? (
                            <AlertCircle className="w-5 h-5 text-amber-400" />
                          ) : (
                            <Sparkles className="w-5 h-5" />
                          )}
                        </div>
                        <div>
                          <h4
                            className={`text-sm font-extrabold font-mono uppercase tracking-wider ${analysisResult?.diagnosis?.crop_compatibility_status === 'MISMATCHED_HOST' || naclRecs.safety_disclaimer?.includes('CAUTION')
                                ? 'text-amber-300'
                                : 'text-emerald-300'
                              }`}
                          >
                            {analysisResult?.diagnosis?.crop_compatibility_status === 'MISMATCHED_HOST'
                              ? 'Host Caution & Advisory'
                              : 'Gemini AI Agronomist Advisory'}
                          </h4>
                          <p className="text-[11px] text-industrial-400 font-mono">
                            Personalized chemical action guide for {naclRecs.crop_name} protection
                          </p>
                        </div>
                      </div>
                    </div>

                    <div
                      className={`p-3.5 rounded-xl text-xs leading-relaxed font-medium border ${analysisResult?.diagnosis?.crop_compatibility_status === 'MISMATCHED_HOST' || naclRecs.safety_disclaimer?.includes('CAUTION')
                          ? 'bg-amber-950/80 border-amber-700/80 text-amber-100'
                          : 'bg-industrial-900/90 border-industrial-700 text-industrial-100'
                        }`}
                    >
                      {naclRecs.ai_advisory_summary}
                    </div>

                    {/* Active Ingredients Chips */}
                    {naclRecs.recommended_active_ingredients.length > 0 && (
                      <div className="flex items-center space-x-2 flex-wrap gap-2 pt-0.5">
                        <span className="text-xs font-mono font-extrabold text-emerald-400 uppercase">
                          Target Chemistry:
                        </span>
                        {naclRecs.recommended_active_ingredients.map((ai, idx) => (
                          <span
                            key={idx}
                            className="px-2.5 py-0.5 rounded-lg bg-emerald-900/60 border border-emerald-700/60 text-emerald-200 text-xs font-mono font-bold"
                          >
                            {ai}
                          </span>
                        ))}
                      </div>
                    )}
                  </Card>

                  {/* Multi-Column NACL Product Cards Grid */}
                  {naclRecs.recommendations.length > 0 && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {naclRecs.recommendations.map((prod, idx) => (
                        <Card
                          key={idx}
                          className="p-4 bg-white border border-industrial-200 hover:border-emerald-500 transition-all shadow-sm hover:shadow-md space-y-3 flex flex-col justify-between"
                        >
                          <div className="space-y-2.5">
                            {/* Title & Category Row */}
                            <div className="flex items-start justify-between gap-2 border-b border-industrial-100 pb-2">
                              <div>
                                <h4 className="text-base font-extrabold text-industrial-900">
                                  {prod.product_name}
                                </h4>
                                <span className="px-2 py-0.5 rounded bg-industrial-100 text-industrial-800 text-[10px] font-mono font-bold uppercase">
                                  {prod.category}
                                </span>
                              </div>

                              <span className="px-2 py-0.5 rounded text-[9px] font-mono font-extrabold border bg-emerald-100 text-emerald-800 border-emerald-300">
                                NACL PRODUCT
                              </span>
                            </div>

                            {/* Active Ingredient & FRAC Group */}
                            {(prod.active_ingredient || prod.frac_group) && (
                              <div className="flex items-center justify-between gap-2 p-2 rounded-lg bg-emerald-50/80 border border-emerald-200/60 text-xs font-mono text-emerald-800 font-extrabold">
                                <span className="truncate">Active: {prod.active_ingredient || 'Formulated Agrochemical'}</span>
                                {prod.frac_group && (
                                  <span className="px-1.5 py-0.5 rounded bg-emerald-200/80 text-emerald-900 text-[9px] shrink-0">
                                    {prod.frac_group}
                                  </span>
                                )}
                              </div>
                            )}

                            {/* Rationale */}
                            <p className="text-xs text-industrial-700 bg-industrial-50 p-2.5 rounded-xl border border-industrial-100 leading-relaxed font-medium">
                              {prod.match_rationale}
                            </p>

                            {/* Dosage & Pack Sizes */}
                            <div className="space-y-1.5 text-xs font-mono">
                              <div className="p-2 rounded-xl bg-industrial-50 border border-industrial-200 space-y-0.5">
                                <div className="flex items-center justify-between text-[9px]">
                                  <span className="text-industrial-500 font-extrabold uppercase">
                                    RECOMMENDED DOSAGE
                                  </span>
                                </div>
                                <span className="text-industrial-900 font-bold block text-xs">
                                  {prod.recommended_dosage}
                                </span>
                              </div>

                              {prod.pack_sizes && prod.pack_sizes.length > 0 && (
                                <div className="space-y-0.5">
                                  <span className="text-[9px] text-industrial-500 font-extrabold block uppercase">
                                    PACK SIZES:
                                  </span>
                                  <div className="flex flex-wrap gap-1">
                                    {prod.pack_sizes.map((ps, pidx) => (
                                      <span
                                        key={pidx}
                                        className="px-1.5 py-0.2 rounded bg-industrial-100 text-industrial-800 text-[9px] font-bold"
                                      >
                                        {ps}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>

                          {/* Direct NACL Product Link Button */}
                          <div className="pt-2 border-t border-industrial-100">
                            <a
                              href={prod.product_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="w-full inline-flex items-center justify-center space-x-1.5 py-2 px-3 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-extrabold transition-all shadow-sm"
                            >
                              <span>View NACL Product Details</span>
                              <ExternalLink className="w-3.5 h-3.5" />
                            </a>
                          </div>
                        </Card>
                      ))}
                    </div>
                  )}

                  {/* Safety Disclaimer */}
                  <div className="p-3.5 rounded-xl bg-industrial-100 border border-industrial-200 text-xs text-industrial-600 flex items-start space-x-2.5 font-mono">
                    <ShieldCheck className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
                    <span>{naclRecs.safety_disclaimer}</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Full History Modal Dialog */}
      {showHistoryModal && (
        <div className="fixed inset-0 z-50 bg-industrial-950/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl border border-industrial-200 shadow-2xl w-full max-w-4xl max-h-[85vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95">
            {/* Modal Header */}
            <div className="p-5 bg-gradient-to-r from-emerald-950 to-industrial-900 text-white flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="p-2 rounded-xl bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                  <History className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-lg font-extrabold tracking-tight">
                    Leaf Pathology Analysis Run History
                  </h3>
                  <p className="text-xs text-industrial-300 font-mono">
                    All historical diagnostic records, pathogen evidence, and NACL recommendations
                  </p>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                {historyList.length > 0 && (
                  <button
                    type="button"
                    onClick={handleClearHistory}
                    className="px-3 py-1.5 rounded-lg bg-reject-900/60 hover:bg-reject-800 text-reject-200 border border-reject-600/50 text-xs font-mono font-bold flex items-center space-x-1.5 transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Clear All History</span>
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => setShowHistoryModal(false)}
                  className="p-1.5 rounded-xl bg-industrial-800 hover:bg-industrial-700 text-white transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto flex-1 space-y-4">
              {isLoadingHistory ? (
                <div className="py-12 text-center text-sm font-mono text-industrial-500 flex items-center justify-center space-x-2">
                  <RefreshCw className="w-4 h-4 animate-spin text-emerald-600" />
                  <span>Fetching analysis history from database...</span>
                </div>
              ) : historyList.length === 0 ? (
                <div className="py-12 text-center text-sm text-industrial-500 font-mono">
                  No historical analysis runs found in database.
                </div>
              ) : (
                <div className="space-y-3">
                  {historyList.map((run) => {
                    const isSelected = activeHistoryId === run.analysis_id || analysisResult?.analysis_id === run.analysis_id;
                    const recCount = run.nacl_recommendations?.recommendations?.length || 0;
                    return (
                      <div
                        key={run.analysis_id}
                        onClick={() => handleSelectHistoryRun(run)}
                        className={`p-4 rounded-xl border transition-all cursor-pointer space-y-2.5 ${isSelected
                            ? 'bg-emerald-50/80 border-emerald-500 ring-2 ring-emerald-500/20'
                            : 'bg-white border-industrial-200 hover:border-emerald-400 hover:shadow-md'
                          }`}
                      >
                        <div className="flex items-center justify-between flex-wrap gap-2">
                          <div className="flex items-center space-x-3">
                            <span className="font-mono text-sm font-extrabold text-industrial-900">
                              {run.analysis_id}
                            </span>
                            <span
                              className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-bold uppercase ${run.status === 'SUCCESS'
                                  ? 'bg-pass-100 text-pass-800 border border-pass-300'
                                  : run.status === 'UNCERTAIN'
                                    ? 'bg-amber-100 text-amber-800 border border-amber-300'
                                    : 'bg-reject-100 text-reject-800 border border-reject-300'
                                }`}
                            >
                              {run.status}
                            </span>
                          </div>

                          <div className="flex items-center space-x-3 text-xs font-mono text-industrial-500">
                            <span className="flex items-center space-x-1">
                              <Clock className="w-3.5 h-3.5" />
                              <span>{new Date(run.timestamp).toLocaleString()}</span>
                            </span>
                            <span className="px-2 py-1 rounded bg-emerald-100 text-emerald-800 text-[10px] font-bold flex items-center space-x-1 hover:bg-emerald-200">
                              <span>Open in New Tab</span>
                              <ExternalLink className="w-3 h-3" />
                            </span>
                            <button
                              type="button"
                              onClick={(e) => handleDeleteRun(e, run.analysis_id)}
                              className="p-1 rounded hover:bg-reject-50 text-industrial-400 hover:text-reject-600 transition-colors"
                              title="Delete record"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono pt-1">
                          <div className="p-2.5 rounded-lg bg-industrial-50 border border-industrial-100">
                            <span className="text-industrial-400 text-[10px] block font-bold">CROP SPECIES</span>
                            <span className="font-extrabold text-industrial-900 text-xs">
                              {run.crop?.crop_name || 'Unknown'}
                            </span>
                          </div>
                          <div className="p-2.5 rounded-lg bg-industrial-50 border border-industrial-100">
                            <span className="text-industrial-400 text-[10px] block font-bold">PATHOLOGY</span>
                            <span className="font-extrabold text-industrial-900 text-xs">
                              {run.disease?.disease_name || 'No pathology'}
                            </span>
                          </div>
                          <div className="p-2.5 rounded-lg bg-industrial-50 border border-industrial-100">
                            <span className="text-industrial-400 text-[10px] block font-bold">NACL RECOMMENDATIONS</span>
                            <span className="font-extrabold text-emerald-700 text-xs">
                              {recCount} Agrochemical Products
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* NACL Agrochemical Catalog Modal Dialog */}
      {showCatalogModal && (
        <div className="fixed inset-0 z-50 bg-industrial-950/75 backdrop-blur-md flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl border border-industrial-200 shadow-2xl w-full max-w-5xl max-h-[88vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95">
            {/* Catalog Modal Header */}
            <div className="p-5 bg-gradient-to-r from-emerald-950 via-industrial-950 to-industrial-900 text-white flex items-center justify-between">
              <div className="flex items-center space-x-3.5">
                <div className="p-2.5 rounded-xl bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                  <Building2 className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-lg font-extrabold tracking-tight flex items-center space-x-2">
                    <span>NACL Industries Agrochemical Catalog</span>
                    <span className="px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 text-xs font-mono">
                      {catalogData ? `${catalogData.total_count} Products Indexed` : '60 Products Indexed'}
                    </span>
                  </h3>
                  <p className="text-xs text-industrial-300 font-mono">
                    Categorized Crop Protection Products: Fungicides, Insecticides, Herbicides & Plant Health Solutions
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={() => setShowCatalogModal(false)}
                className="p-2 rounded-xl bg-industrial-800 hover:bg-industrial-700 text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Catalog Controls: Filter Categories & Search Bar */}
            <div className="p-4 bg-industrial-50 border-b border-industrial-200 space-y-3">
              <div className="flex items-center justify-between flex-wrap gap-3">
                {/* Categories Tabs */}
                <div className="flex items-center space-x-2 overflow-x-auto pb-1 scrollbar-none">
                  {['All', 'Fungicides', 'Insecticides', 'Herbicides', 'Plant Health'].map((cat) => (
                    <button
                      key={cat}
                      type="button"
                      onClick={() => setCatalogActiveCategory(cat)}
                      className={`px-3 py-1.5 rounded-xl text-xs font-mono font-bold transition-all whitespace-nowrap ${catalogActiveCategory === cat
                          ? 'bg-emerald-700 text-white shadow-sm'
                          : 'bg-white border border-industrial-200 text-industrial-700 hover:bg-emerald-50 hover:border-emerald-300'
                        }`}
                    >
                      {cat}
                    </button>
                  ))}
                </div>

                {/* Search Bar */}
                <div className="relative flex-1 min-w-[220px] max-w-xs">
                  <Search className="w-4 h-4 text-industrial-400 absolute left-3 top-2.5" />
                  <input
                    type="text"
                    value={catalogSearch}
                    onChange={(e) => setCatalogSearch(e.target.value)}
                    placeholder="Search product, ingredient, or crop..."
                    className="w-full pl-9 pr-3 py-1.5 rounded-xl border border-industrial-300 text-xs font-mono text-industrial-900 focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                  />
                </div>
              </div>
            </div>

            {/* Catalog Modal Body */}
            <div className="p-6 overflow-y-auto flex-1 space-y-4">
              {isLoadingCatalog ? (
                <div className="py-16 text-center text-xs font-mono text-industrial-400 animate-pulse flex items-center justify-center space-x-2">
                  <RefreshCw className="w-4 h-4 animate-spin text-emerald-600" />
                  <span>Loading official NACL agrochemical database...</span>
                </div>
              ) : !catalogData ? (
                <div className="py-16 text-center text-xs font-mono text-industrial-400">
                  Unable to load NACL catalog. Please ensure backend is running.
                </div>
              ) : (
                <div className="space-y-6">
                  {Object.entries(catalogData.categories)
                    .filter(([category]) => catalogActiveCategory === 'All' || category.toLowerCase().includes(catalogActiveCategory.toLowerCase()))
                    .map(([category, products]) => {
                      const filteredProds = products.filter((p: any) => {
                        if (!catalogSearch.trim()) return true;
                        const q = catalogSearch.toLowerCase();
                        return (
                          p.product_name?.toLowerCase().includes(q) ||
                          p.active_ingredient?.toLowerCase().includes(q) ||
                          p.mode_of_action?.toLowerCase().includes(q)
                        );
                      });

                      if (filteredProds.length === 0) return null;

                      return (
                        <div key={category} className="space-y-3">
                          <div className="flex items-center space-x-2 border-b border-industrial-200 pb-2">
                            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                            <h4 className="text-sm font-extrabold text-industrial-900 font-mono uppercase tracking-wider">
                              {category} ({filteredProds.length})
                            </h4>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {filteredProds.map((prod: any) => (
                              <div
                                key={prod.product_id}
                                className="p-4 rounded-xl border border-industrial-200 bg-white hover:border-emerald-400 hover:shadow-md transition-all space-y-2.5 flex flex-col justify-between"
                              >
                                <div className="space-y-2">
                                  <div className="flex items-start justify-between gap-2">
                                    <h5 className="font-extrabold text-industrial-900 text-sm">
                                      {prod.product_name}
                                    </h5>
                                    {prod.product_url && (
                                      <a
                                        href={prod.product_url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-emerald-600 hover:text-emerald-800 transition-colors p-1"
                                        title="View official NACL product page"
                                      >
                                        <ExternalLink className="w-4 h-4" />
                                      </a>
                                    )}
                                  </div>

                                  {prod.active_ingredient && (
                                    <p className="text-[11px] font-mono font-bold text-emerald-800 bg-emerald-50 p-1.5 rounded border border-emerald-200">
                                      Active: {prod.active_ingredient}
                                    </p>
                                  )}

                                  {prod.mode_of_action && (
                                    <p className="text-xs text-industrial-600 line-clamp-3 leading-relaxed font-medium">
                                      {prod.mode_of_action}
                                    </p>
                                  )}
                                </div>

                                {prod.pack_sizes && prod.pack_sizes.length > 0 && (
                                  <div className="pt-2 border-t border-industrial-100 flex items-center justify-between text-[10px] font-mono text-industrial-500">
                                    <span>Packs: {prod.pack_sizes.join(', ')}</span>
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default LeafDiseasePage;
