import React from 'react';
import { Sparkles, RefreshCw, AlertCircle, CheckCircle, Play } from 'lucide-react';
import { VLMAnalysis } from '../../types';
import { Badge } from '../common/Badge';
import { Button } from '../common/Button';

interface VLMDefectAnalysisCardProps {
  isPass: boolean;
  vlmAnalysis?: VLMAnalysis | null;
  onGenerate?: (force?: boolean) => void;
  isGenerating?: boolean;
}

export const VLMDefectAnalysisCard: React.FC<VLMDefectAnalysisCardProps> = ({
  isPass,
  vlmAnalysis,
  onGenerate,
  isGenerating = false,
}) => {
  const getSeverityBadgeType = (severity?: string): 'reject' | 'review' | 'info' => {
    switch ((severity || '').toLowerCase()) {
      case 'critical':
      case 'high':
        return 'reject';
      case 'medium':
        return 'review';
      case 'low':
        return 'info';
      default:
        return 'info';
    }
  };

  if (isPass) {
    return (
      <div className="pt-4 border-t border-industrial-200 mt-4">
        <div className="flex items-center space-x-2 text-industrial-900 font-bold mb-2">
          <Sparkles className="w-4 h-4 text-accent-500" />
          <h4 className="text-xs font-bold uppercase tracking-wider text-industrial-600">AI DEFECT ANALYSIS</h4>
          <span className="text-xs bg-industrial-100 text-industrial-700 px-2 py-0.5 rounded font-medium">
            Powered by Gemini
          </span>
        </div>
        <div className="p-3 bg-industrial-50 rounded-lg border border-industrial-200 flex items-center space-x-2 text-industrial-700 text-sm">
          <CheckCircle className="w-4 h-4 text-pass-500 shrink-0" />
          <span>Not required — PatchCore classified this image as normal.</span>
        </div>
      </div>
    );
  }

  const status = vlmAnalysis?.status || 'not_generated';
  const isLoading = isGenerating || status === 'generating';

  return (
    <div className="pt-4 border-t border-industrial-200 mt-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="p-1 rounded bg-brand-50 text-brand-600 border border-brand-200">
            <Sparkles className="w-4 h-4 text-brand-600" />
          </div>
          <h4 className="text-xs font-extrabold uppercase tracking-wider text-industrial-900 flex items-center gap-1.5">
            AI DEFECT ANALYSIS
          </h4>
          <span className="text-xs bg-brand-50 text-brand-800 border border-brand-200 px-2 py-0.5 rounded font-bold">
            Powered by Gemini
          </span>
        </div>

        {/* Retry / Regenerate Button for Failed or Unavailable states */}
        {onGenerate && (status === 'unavailable' || status === 'failed') && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => onGenerate(true)}
            disabled={isLoading}
            className="text-xs h-7 px-2.5 py-0.5 font-bold"
          >
            <RefreshCw className={`w-3.5 h-3.5 mr-1 ${isLoading ? 'animate-spin' : ''}`} />
            Retry
          </Button>
        )}
      </div>

      {/* 1. Un-generated initial state (Show prominent Generate button) */}
      {!vlmAnalysis && status === 'not_generated' && !isLoading && (
        <div className="p-4 bg-industrial-50 rounded-xl border border-industrial-200 text-center space-y-3">
          <p className="text-sm text-industrial-700 font-semibold">
            PatchCore flagged an anomaly. Click below to analyze visual evidence with Google Gemini AI.
          </p>
          {onGenerate && (
            <Button
              variant="primary"
              size="sm"
              onClick={() => onGenerate(false)}
              icon={<Play className="w-3.5 h-3.5" />}
              className="mx-auto text-xs sm:text-sm font-bold"
            >
              Generate AI Defect Analysis
            </Button>
          )}
        </div>
      )}

      {/* 2. Generating / Loading state */}
      {isLoading && (
        <div className="p-4 bg-slate-900 text-white rounded-xl border border-slate-800 flex items-center justify-center space-x-3 text-sm">
          <RefreshCw className="w-4 h-4 text-accent-400 animate-spin shrink-0" />
          <span className="font-semibold text-slate-200">Analyzing visual evidence with Gemini AI...</span>
        </div>
      )}

      {/* 3. Completed State (Structured Card) */}
      {status === 'completed' && vlmAnalysis && !isLoading && (
        <div className="bg-slate-900 text-white rounded-xl p-4 space-y-3.5 border border-slate-800 text-sm">
          <div className="grid grid-cols-2 gap-3 pb-2.5 border-b border-slate-800">
            <div>
              <span className="text-slate-400 block text-xs uppercase font-bold tracking-wide">Defect Type</span>
              <span className="font-bold text-white text-sm">
                {vlmAnalysis.defect_type || 'Unknown'}
              </span>
            </div>
            <div>
              <span className="text-slate-400 block text-xs uppercase font-bold tracking-wide">Location</span>
              <span className="font-bold text-white text-sm">
                {vlmAnalysis.location || 'Unspecified'}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 pb-2.5 border-b border-slate-800">
            <div>
              <span className="text-slate-400 block text-xs uppercase font-bold tracking-wide">Severity</span>
              <div className="mt-1">
                <Badge type={getSeverityBadgeType(vlmAnalysis.severity)}>
                  {vlmAnalysis.severity || 'Medium'}
                </Badge>
              </div>
            </div>
            <div>
              <span className="text-slate-400 block text-xs uppercase font-bold tracking-wide">Prominence</span>
              <span className="font-bold text-slate-200 text-sm">
                {vlmAnalysis.prominence || 'Moderate'}
              </span>
            </div>
          </div>

          {vlmAnalysis.visual_evidence && (
            <div>
              <span className="text-slate-400 block text-xs uppercase font-bold tracking-wide mb-1">Visual Evidence</span>
              <p className="text-slate-200 text-xs sm:text-sm leading-relaxed bg-slate-950/70 p-3 rounded-lg border border-slate-800/90 font-medium">
                {vlmAnalysis.visual_evidence}
              </p>
            </div>
          )}

          {vlmAnalysis.explanation && (
            <div>
              <span className="text-slate-400 block text-xs uppercase font-bold tracking-wide mb-1">Explanation</span>
              <p className="text-slate-200 text-xs sm:text-sm leading-relaxed bg-slate-950/70 p-3 rounded-lg border border-slate-800/90 font-medium">
                {vlmAnalysis.explanation}
              </p>
            </div>
          )}

          {vlmAnalysis.confidence !== undefined && vlmAnalysis.confidence !== null && (
            <div className="flex items-center justify-between text-xs text-slate-400 pt-1 font-mono">
              <span className="font-semibold text-slate-400">Interpretation Confidence:</span>
              <span className="text-accent-400 font-extrabold text-sm">
                {(vlmAnalysis.confidence * 100).toFixed(0)}%
              </span>
            </div>
          )}
        </div>
      )}

      {/* 4. Unavailable State */}
      {status === 'unavailable' && !isLoading && (
        <div className="p-4 bg-amber-50 rounded-xl border border-amber-200 text-sm space-y-2">
          <div className="flex items-center space-x-1.5 text-amber-900 font-bold">
            <AlertCircle className="w-4.5 h-4.5 text-amber-600 shrink-0" />
            <span>Analysis Unavailable</span>
          </div>
          <p className="text-amber-800 text-xs sm:text-sm pl-6 leading-relaxed font-medium">
            {vlmAnalysis?.explanation || 'GEMINI_API_KEY is not configured or service is temporarily unreachable.'}
          </p>
          {onGenerate && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onGenerate(true)}
              className="text-xs h-7 px-3 py-1 ml-6 font-bold"
            >
              <RefreshCw className="w-3.5 h-3.5 mr-1" />
              Generate AI Defect Analysis Again
            </Button>
          )}
        </div>
      )}

      {/* 5. Failed State */}
      {status === 'failed' && !isLoading && (
        <div className="p-4 bg-reject-50 rounded-xl border border-reject-200 text-sm space-y-2">
          <div className="flex items-center space-x-1.5 text-reject-900 font-bold">
            <AlertCircle className="w-4.5 h-4.5 text-reject-600 shrink-0" />
            <span>Analysis Failed</span>
          </div>
          <p className="text-reject-800 text-xs sm:text-sm pl-6 leading-relaxed font-medium">
            {vlmAnalysis?.explanation || 'Gemini VLM analysis encountered an error while processing.'}
          </p>
          {onGenerate && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onGenerate(true)}
              className="text-xs h-7 px-3 py-1 ml-6 font-bold"
            >
              <RefreshCw className="w-3.5 h-3.5 mr-1" />
              Generate AI Defect Analysis Again
            </Button>
          )}
        </div>
      )}
    </div>
  );
};
