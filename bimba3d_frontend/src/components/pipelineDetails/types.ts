export interface PipelineDetail {
  id: string;
  name: string;
  status: string;
  pipeline_type?: string;
  workflow_stage?: string;
  created_at: string;
  updated_at?: string | null;
  started_at: string | null;
  completed_at: string | null;
  current_phase: number;
  current_run: number;
  current_project_index: number;
  total_runs: number;
  completed_runs: number;
  failed_runs: number;
  hard_cap_runs?: number;
  pending_runs?: number;
  mean_relative_score: number | null;
  success_rate: number | null;
  best_relative_score: number | null;
  last_error: string | null;
  cooldown_active: boolean;
  next_run_scheduled_at: string | null;
  current_test_model_id?: string | null;
  config?: any;
  runs?: any[];
  active_run?: {
    project_name?: string | null;
    run_id?: string | null;
    phase?: number | null;
    run?: number | null;
    test_model_id?: string | null;
    status?: string | null;
    started_at?: string | null;
    selected_preset?: string | null;
    selected_multipliers?: Record<string, number> | null;
    selected_multipliers_raw?: Record<string, number> | null;
    selected_log_multipliers?: Record<string, number> | null;
    candidate_score_checks?: Record<string, unknown> | null;
    initial_params?: Record<string, number> | null;
  } | null;
  latest_prediction_preview_key?: string | null;
  prediction_previews?: Record<string, any> | null;
}

export interface DetailTab {
  id: string;
  label: string;
}

