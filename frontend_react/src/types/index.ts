export type ModelStatus = 'active' | 'inactive' | 'draft' | 'building' | 'error';

export interface ModelGroup {
  id: string;
  _id?: string;
  name: string;
  description?: string | null;
  model_count?: number;
  models?: Model[];
  created_at?: string;
  updated_at?: string;
}

export interface Model {
  id: string;
  _id?: string;
  name: string;
  description?: string | null;
  status: ModelStatus;
  domain?: string | null;
  group_id?: string | null;
  group?: ModelGroup | null;
  error_reason?: string | null;
  active_version_id?: string | null;
  reference_image_count: number;
  created_at: string;
  updated_at: string;
}

export interface ModelVersion {
  id: string;
  _id?: string;
  model_id: string;
  version_number: number;
  status: 'ready' | 'failed' | 'building';
  algorithm?: {
    backbone: string;
    layers: string[];
    coreset_sampling_ratio: number;
  };
  calibration?: {
    threshold: number;
    method?: string;
    auto_calibrated_threshold?: number | null;
    is_custom?: boolean;
  };
  training?: {
    build_time_ms: number;
    reference_count: number;
  };
  artifacts?: {
    checkpoint_uri: string;
    memory_bank_uri: string;
    version_metadata_uri: string;
  };
  created_at: string;
}

export interface ReferenceImage {
  id: string;
  _id?: string;
  model_id: string;
  version_id?: string | null;
  filename: string;
  storage_uri?: string;
  file_size_bytes?: number;
  checksum?: string;
  created_at?: string;
  storage?: {
    uri?: string;
  };
  relative_path?: string;
  file_size?: number;
  uploaded_at?: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PredictionOutput {
  status: 'normal' | 'anomalous' | 'PASS' | 'REJECT';
  anomaly_score: number;
  threshold: number;
  severity: 'none' | 'low' | 'medium' | 'high' | 'critical';
}

export interface LocalizationOutput {
  bbox: BoundingBox | null;
  heatmap_uri: string | null;
}

export interface RunSummary {
  total: number;
  pass: number;
  reject: number;
  errors: number;
}

export interface InspectionRun {
  id: string;
  _id?: string;
  run_id?: string;
  model_id: string;
  model_version_id: string;
  run_number: number;
  status: 'running' | 'completed' | 'partial' | 'failed';
  total_images: number;
  completed_images: number;
  pass_count: number;
  reject_count: number;
  error_count: number;
  summary: RunSummary;
  created_at: string;
  completed_at?: string | null;
  inspections?: Inspection[];
}

export interface VLMAnalysis {
  status: 'completed' | 'unavailable' | 'failed' | 'skipped' | 'not_generated' | 'generating';
  provider: string;
  model?: string;
  defect_type?: string;
  location?: string;
  severity?: 'Low' | 'Medium' | 'High' | 'Critical' | string;
  prominence?: string;
  explanation?: string;
  visual_evidence?: string;
  confidence?: number;
  generated_at?: string;
}

export interface OverallPrediction {
  status: 'PASS' | 'REJECT' | 'REVIEW' | string;
  total_instances: number;
  pass_count: number;
  reject_count: number;
  error_count?: number;
  max_anomaly_score?: number | null;
  threshold?: number | null;
  reason?: 'NO_OBJECTS_DETECTED' | 'INSTANCE_DETECTION_FAILED' | 'PATCHCORE_INSTANCE_FAILED' | string | null;
  message?: string | null;
}

export interface InstanceResult {
  instance_id: number;
  bbox: BoundingBox;
  padded_bbox?: BoundingBox;
  crop_storage_uri: string;
  detection_confidence?: number;
  status: 'completed' | 'failed' | string;
  error?: string | null;
  prediction?: PredictionOutput | null;
  localization?: LocalizationOutput | null;
  vlm_analysis?: VLMAnalysis | null;
}

export interface Inspection {
  id: string;
  _id?: string;
  inspection_id?: string;
  model_id: string;
  model_version_id: string;
  run_id?: string | null;
  run_number?: number | null;
  inspection_mode?: 'single' | 'multi_instance';
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'review';
  processing_time_ms?: number;
  filename?: string;
  storage_uri?: string;
  input?: {
    filename?: string;
    storage_uri?: string;
  };
  created_at: string;
  prediction?: PredictionOutput;
  overall_prediction?: OverallPrediction;
  localization?: LocalizationOutput;
  instances?: InstanceResult[];
  composite_heatmap_uri?: string | null;
  vlm_analysis?: VLMAnalysis;
  processing_stats?: {
    detection_time_ms?: number;
    patchcore_time_ms?: number;
    total_time_ms?: number;
    instance_count?: number;
  };
}

export interface InspectionResultResponse {
  inspection_id: string;
  model_id: string;
  model_version_id: string;
  inspection_mode?: 'single' | 'multi_instance';
  status: string;
  filename: string;
  storage_uri: string;
  processing_time_ms: number;
  prediction?: PredictionOutput;
  overall_prediction?: OverallPrediction;
  localization?: LocalizationOutput;
  instances?: InstanceResult[];
  composite_heatmap_uri?: string | null;
  vlm_analysis?: VLMAnalysis;
  processing_stats?: {
    detection_time_ms?: number;
    patchcore_time_ms?: number;
    total_time_ms?: number;
    instance_count?: number;
  };
  created_at?: string;
}

export interface Feedback {
  id: string;
  _id?: string;
  inspection_id: string;
  inspection_result_id?: string | null;
  model_id: string;
  model_version_id: string;
  run_id?: string | null;
  original_prediction?: PredictionOutput;
  original_vlm_analysis?: VLMAnalysis;
  detection_feedback: 'correct' | 'false_positive' | 'false_negative';
  vlm_feedback_categories: ('correct' | 'wrong_defect_type' | 'wrong_location' | 'wrong_severity')[];
  corrected_defect_type?: string | null;
  corrected_location?: string | null;
  corrected_severity?: 'Low' | 'Medium' | 'High' | 'Critical' | string | null;
  comment?: string | null;
  feedback_type?: string;
  created_at: string;
  filename?: string;
  storage_uri?: string;
  prediction?: PredictionOutput;
  vlm_analysis?: VLMAnalysis;
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  database: {
    status: string;
    name: string;
  };
  cuda: {
    available: boolean;
    device_count?: number;
    device_name?: string;
  };
  patchcore: {
    engine: string;
    anomalib_available: boolean;
    model_cache_size: number;
  };
}

export interface InspectionBatchItem {
  id: string;
  file: File;
  previewUrl: string;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  result?: InspectionResultResponse;
  error?: string;
}
