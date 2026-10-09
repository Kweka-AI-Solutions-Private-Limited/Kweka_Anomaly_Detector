# Anomaly Detector — Research & Experiments

This directory contains **research-only** scripts for offline ML experiments,
evaluation runs, ablation studies, and visualization utilities.

> **These files are NOT part of the running application.**
> They are excluded from the production Docker image.
> Do NOT import these from `src/` application code.

## Contents

| Script | Purpose |
|---|---|
| evaluate_draem.py | DRAEM baseline evaluation |
| evaluate_efficientad.py | EfficientAD baseline evaluation |
| evaluate_patchcore.py | PatchCore general evaluation |
| evaluate_patchcore_l1l2.py | Layer 1+2 ablation evaluation |
| evaluate_patchcore_layer1.py | Layer 1 only ablation |
| evaluate_patchcore_per_defect.py | Per-defect class analysis |
| evaluate_rd4ad.py | RD4AD baseline evaluation |
| evaluate_stfpm.py | STFPM baseline evaluation |
| train_draem.py | DRAEM training |
| train_efficientad.py | EfficientAD training |
| train_rd4ad.py | RD4AD training |
| train_rd4ad_direct.py | RD4AD direct training variant |
| train_rd4ad_fast.py | RD4AD fast training variant |
| train_stfpm.py | STFPM training |
| run_patchcore_baseline.py | PatchCore baseline runner |
| run_patchcore_generic.py | Generic PatchCore runner |
| run_patchcore_l1l2.py | Layer ablation runner |
| run_patchcore_layer1.py | Layer 1 runner |
| run_patchcore_failure_analysis.py | Failure analysis |
| run_patchcore_technical_audit.py | Technical audit runner |
| run_pcb_transistor_technical_audit.py | PCB/transistor audit |
| run_positional_encoding_experiment.py | Positional encoding experiment |
| run_textile_250_30_calibration_eval.py | Textile calibration (250/30 split) |
| run_textile_280_loo_eval.py | Textile leave-one-out evaluation |
| run_textile_good_augmentation_eval.py | Textile augmentation experiment |
| run_dtd_patchcore_demo.py | DTD texture demo |
| run_localization_experiment.py | Localization experiment |
| analyze_layer1_sensitivity.py | Layer 1 sensitivity analysis |
| build_layer1_checkpoint.py | Layer 1 checkpoint builder |
| diagnose_textile_calibration.py | Textile calibration diagnostic |
| measure_patchcore_operational.py | Operational performance measurement |
| optimize_screw_localization.py | Screw localization optimization |
| demo_patchcore_visualization.py | PatchCore visual demo |
| visualize_patchcore_results.py | Results visualization |
| visualize_textile_false_positives.py | False positive visualization |
| format_textile_report.py | Textile report formatting |
| print_summary.py | Summary printer utility |
| debug_patchcore.py | PatchCore debug utility |
| localization_experiment.py | Localization service experiment stub |
| positional_encoding_experiment.py | Positional encoding experiment stub |
| test_diagnostic_patchcore.py | Diagnostic test script |
| test_localization_exp2.py | Localization test experiment 2 |
| test_textile_pcb_localization.py | Textile PCB localization test |

## Running Experiments

Install development dependencies first:

```bash
pip install -r requirements.dev.txt
```

Run from the `backend/` directory so relative paths to `data/` resolve correctly.

```bash
cd backend/
python experiments/run_patchcore_baseline.py
```
