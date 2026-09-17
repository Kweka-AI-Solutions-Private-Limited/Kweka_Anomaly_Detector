import { apiClient } from './client';

export interface DashboardKPI {
  total_inspections: number;
  pass_count: number;
  reject_count: number;
  error_count: number;
  pass_rate: number;
  total_runs: number;
  active_models: number;
}

export interface PassVsReject {
  pass_count: number;
  reject_count: number;
  pass_percentage: number;
  reject_percentage: number;
}

export interface DailyTrendItem {
  date: string;
  total: number;
  pass: number;
  reject: number;
}

export interface DefectDistributionItem {
  defect_type: string;
  count: number;
}

export interface ModelDistributionItem {
  model_name: string;
  count: number;
}

export interface RecentRunSummaryItem {
  id: string;
  run_number: number;
  model_name: string;
  version_number: number;
  total_images: number;
  status: string;
}

export interface RunSummary {
  total_runs: number;
  completed_runs: number;
  failed_runs: number;
  recent_runs: RecentRunSummaryItem[];
}

export interface FeedbackSummaryCategories {
  correct: number;
  false_positive: number;
  false_negative: number;
  wrong_defect_type: number;
  wrong_location: number;
  wrong_severity: number;
}

export interface FeedbackSummary {
  total_feedback: number;
  categories: FeedbackSummaryCategories;
}

export interface DashboardRecentInspection {
  id: string;
  filename: string;
  storage_uri?: string;
  model_name: string;
  model_version_number?: number;
  run_number?: number;
  run_id?: string;
  verdict: string;
  anomaly_score?: number;
  defect_type: string;
  severity: string;
  created_at: string;
}

export interface HotspotCell {
  row: number;
  col: number;
  count: number;
  intensity: number;
}

export interface AnomalyHotspotAnalysis {
  grid_size: number;
  total_anomalous_inspections: number;
  inspections_with_bbox: number;
  matrix: number[][];
  max_cell_count: number;
  cells: HotspotCell[];
}

export interface DashboardSummaryResponse {
  kpi: DashboardKPI;
  pass_vs_reject: PassVsReject;
  inspection_trend: DailyTrendItem[];
  defect_distribution: DefectDistributionItem[];
  anomaly_hotspot_analysis: AnomalyHotspotAnalysis;
  severity_distribution: Record<string, number>;
  model_distribution: ModelDistributionItem[];
  run_summary: RunSummary;
  feedback_summary: FeedbackSummary;
  recent_inspections: DashboardRecentInspection[];
}

export async function getDashboardSummary(params?: {
  model_id?: string;
  start_date?: string;
  end_date?: string;
}): Promise<DashboardSummaryResponse> {
  const queryParams = new URLSearchParams();
  if (params?.model_id && params.model_id !== 'all') queryParams.append('model_id', params.model_id);
  if (params?.start_date) queryParams.append('start_date', params.start_date);
  if (params?.end_date) queryParams.append('end_date', params.end_date);

  const url = `/api/dashboard/summary${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
  const response = await apiClient.get<DashboardSummaryResponse>(url);
  return response.data;
}
