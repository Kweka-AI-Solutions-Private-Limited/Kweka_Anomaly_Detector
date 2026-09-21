import { apiClient } from './client';

export interface ImageValidationDetail {
  filename: string;
  is_valid: boolean;
  format?: string;
  width?: number;
  height?: number;
  size_bytes: number;
  blur_score: number;
  brightness: number;
  errors: string[];
  warnings: string[];
}

export interface OverallImageValidationResult {
  total_submitted: number;
  valid_count: number;
  invalid_count: number;
  is_any_valid: boolean;
  details: ImageValidationDetail[];
}

export interface CropIdentificationResult {
  crop_name: string;
  confidence?: number;
  source: string;
  status: 'CONFIRMED' | 'USER_SPECIFIED' | 'UNCERTAIN';
  evidence_note: string;
}

export interface DiseaseCandidate {
  disease_name: string;
  confidence?: number;
}

export interface NormalizedDiseaseResult {
  disease_name: string;
  confidence?: number;
  provider_name: string;
  is_mock: boolean;
  supported: boolean;
  candidates: DiseaseCandidate[];
  provider_status: 'OK' | 'UNSUPPORTED_CROP' | 'LOW_CONFIDENCE' | 'PROVIDER_ERROR' | 'MOCK';
  error_message?: string;
}

export interface DiagnosisValidationResult {
  evidence_level: 'HIGH' | 'MEDIUM' | 'LOW';
  image_quality_status: string;
  crop_compatibility_status: string;
  consistency_status: string;
  is_uncertain: boolean;
  validation_notes: string[];
}

export interface NACLProductRecommendation {
  product_id: string;
  product_name: string;
  category: string;
  product_url: string;
  active_ingredient?: string;
  match_type: string;
  verification_status: string;
  dosage_verified: boolean;
  recommended_dosage: string;
  match_rationale: string;
  verification_reason?: string;
  pack_sizes: string[];
  source_url?: string;
  source_type?: string;
  frac_group?: string;
}

export interface ProductRecommendationResult {
  crop_name: string;
  disease_name: string;
  is_healthy: boolean;
  recommendations: NACLProductRecommendation[];
  recommended_active_ingredients: string[];
  ai_advisory_summary: string;
  safety_disclaimer: string;
}

export interface LeafDiseaseAnalysisResponse {
  analysis_id: string;
  timestamp: string;
  status: 'SUCCESS' | 'UNCERTAIN' | 'FAILED';
  validation: OverallImageValidationResult;
  crop: CropIdentificationResult;
  disease?: NormalizedDiseaseResult;
  diagnosis?: DiagnosisValidationResult;
  nacl_recommendations?: ProductRecommendationResult;
  retry_guidance?: string;
  phase_notice: string;
}

export async function analyzeLeafDisease(
  files: File[],
  selectedCrop?: string
): Promise<LeafDiseaseAnalysisResponse> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  if (selectedCrop && selectedCrop.trim()) {
    formData.append('selected_crop', selectedCrop.trim());
  }

  const response = await apiClient.post<LeafDiseaseAnalysisResponse>(
    '/api/leaf-disease/analyze',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
}

export async function getLeafDiseaseHistory(
  limit: number = 50,
  status?: string,
  cropName?: string
): Promise<LeafDiseaseAnalysisResponse[]> {
  const params: Record<string, any> = { limit };
  if (status) params.status = status;
  if (cropName) params.crop_name = cropName;

  const response = await apiClient.get<LeafDiseaseAnalysisResponse[]>(
    '/api/leaf-disease/history',
    { params }
  );

  return response.data;
}

export async function getLeafDiseaseRun(
  analysisId: string
): Promise<LeafDiseaseAnalysisResponse> {
  const response = await apiClient.get<LeafDiseaseAnalysisResponse>(
    `/api/leaf-disease/history/${analysisId}`
  );
  return response.data;
}

export async function deleteLeafDiseaseRun(
  analysisId: string
): Promise<{ status: string; analysis_id: string }> {
  const response = await apiClient.delete<{ status: string; analysis_id: string }>(
    `/api/leaf-disease/history/${analysisId}`
  );
  return response.data;
}

export async function clearLeafDiseaseHistory(): Promise<{ status: string; deleted_count: number }> {
  const response = await apiClient.delete<{ status: string; deleted_count: number }>(
    '/api/leaf-disease/history'
  );
  return response.data;
}

export async function getNaclCatalog(): Promise<{
  total_count: number;
  categories: Record<string, any[]>;
}> {
  const response = await apiClient.get<{
    total_count: number;
    categories: Record<string, any[]>;
  }>('/api/leaf-disease/catalog');
  return response.data;
}

export async function submitLeafDiseaseFeedback(feedbackData: {
  analysis_id?: string;
  rating?: string;
  feedback_category?: string;
  comments?: string;
  crop_name?: string;
  disease_name?: string;
}): Promise<{ status: string; feedback_id: string; message: string }> {
  const response = await apiClient.post<{ status: string; feedback_id: string; message: string }>(
    '/api/leaf-disease/feedback',
    feedbackData
  );
  return response.data;
}

