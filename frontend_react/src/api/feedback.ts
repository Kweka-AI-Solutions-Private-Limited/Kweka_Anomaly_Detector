import { apiClient } from './client';
import { Feedback } from '../types';

export interface SubmitFeedbackPayload {
  inspectionId: string;
  detection_feedback?: 'correct' | 'false_positive' | 'false_negative';
  vlm_feedback_categories?: ('correct' | 'wrong_defect_type' | 'wrong_location' | 'wrong_severity')[];
  corrected_defect_type?: string | null;
  corrected_location?: string | null;
  corrected_severity?: string | null;
  comment?: string | null;
  feedback_type?: string;
}

export async function submitExpandedFeedback(payload: SubmitFeedbackPayload): Promise<Feedback> {
  const response = await apiClient.post(
    `/api/inspections/${payload.inspectionId}/feedback`,
    {
      detection_feedback: payload.detection_feedback || 'correct',
      vlm_feedback_categories: payload.vlm_feedback_categories || [],
      corrected_defect_type: payload.corrected_defect_type || null,
      corrected_location: payload.corrected_location || null,
      corrected_severity: payload.corrected_severity || null,
      comment: payload.comment || null,
      feedback_type: payload.feedback_type,
    }
  );
  return response.data;
}

export async function submitFeedback(
  inspectionId: string,
  feedbackType: string,
  comment?: string
): Promise<Feedback> {
  return submitExpandedFeedback({
    inspectionId,
    feedback_type: feedbackType,
    comment,
  });
}

export async function getInspectionFeedback(inspectionId: string): Promise<Feedback[]> {
  const response = await apiClient.get(`/api/inspections/${inspectionId}/feedback`);
  return response.data;
}

export async function getWorkspaceFeedback(filters?: {
  model_id?: string;
  model_version_id?: string;
  run_id?: string;
  feedback_type?: string;
}): Promise<Feedback[]> {
  const response = await apiClient.get('/api/inspections/feedback/workspace', {
    params: filters,
  });
  return response.data;
}

