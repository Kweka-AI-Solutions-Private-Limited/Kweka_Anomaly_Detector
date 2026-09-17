import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import test_history
import test_dashboard

def run_all():
    print("=== Running History Tests ===")
    mock_db_hist = test_history.create_mock_db()
    hist_test_funcs = [
        ("test_unique_model_options", test_history.test_unique_model_options),
        ("test_unique_version_options", test_history.test_unique_version_options),
        ("test_versions_scoped_to_selected_model", test_history.test_versions_scoped_to_selected_model),
        ("test_runs_scoped_to_selected_model_and_version", test_history.test_runs_scoped_to_selected_model_and_version),
        ("test_cascading_disabled_states_conceptually", test_history.test_cascading_disabled_states_conceptually),
        ("test_changing_model_resets_version_and_run", test_history.test_changing_model_resets_version_and_run),
        ("test_changing_version_resets_run", test_history.test_changing_version_resets_run),
        ("test_invalid_combination_returns_empty", test_history.test_invalid_combination_returns_empty),
        ("test_multiple_filters_and_clear", test_history.test_multiple_filters_and_clear),
        ("test_legacy_run_id_null_visible", test_history.test_legacy_run_id_null_visible),
        ("test_inspection_detail_navigation", test_history.test_inspection_detail_navigation),
        ("test_feedback_submission_compatibility", test_history.test_feedback_submission_compatibility),
    ]

    passed = 0
    failed = 0
    for name, func in hist_test_funcs:
        try:
            if name == "test_cascading_disabled_states_conceptually":
                func()
            else:
                func(mock_db_hist)
            print(f"✓ {name} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            failed += 1

    print("\n=== Running Dashboard Tests ===")
    dash_test_funcs = [
        ("test_dashboard_returns_kpi_summary", test_dashboard.test_dashboard_returns_kpi_summary),
        ("test_total_inspection_count_is_correct", test_dashboard.test_total_inspection_count_is_correct),
        ("test_pass_and_reject_counts", test_dashboard.test_pass_and_reject_counts),
        ("test_pass_rate_excludes_errors", test_dashboard.test_pass_rate_excludes_errors),
        ("test_model_filter", test_dashboard.test_model_filter),
        ("test_date_filter", test_dashboard.test_date_filter),
        ("test_defect_distribution", test_dashboard.test_defect_distribution),
        ("test_severity_distribution", test_dashboard.test_severity_distribution),
        ("test_model_distribution", test_dashboard.test_model_distribution),
        ("test_run_counts", test_dashboard.test_run_counts),
        ("test_feedback_counts_scoped", test_dashboard.test_feedback_counts_scoped),
        ("test_recent_inspections_ordered", test_dashboard.test_recent_inspections_ordered),
        ("test_legacy_inspections_contribute_and_standalone", test_dashboard.test_legacy_inspections_contribute_and_standalone),
        ("test_no_vlm_data_no_fake_defects", test_dashboard.test_no_vlm_data_no_fake_defects),
        ("test_empty_dashboard_state", test_dashboard.test_empty_dashboard_state),
        ("test_reconciliation_with_history", test_dashboard.test_reconciliation_with_history),
    ]

    for name, func in dash_test_funcs:
        mock_db_dash = test_dashboard.create_mock_db()
        try:
            if name == "test_empty_dashboard_state":
                func()
            else:
                func(mock_db_dash)
            print(f"✓ {name} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\nOverall Results: {passed} PASSED, {failed} FAILED.")

if __name__ == "__main__":
    run_all()
