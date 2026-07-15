from bimba3d_backend.worker.ai_input_modes.relative_quality_score import compute_relative_quality_summary


def test_relative_quality_summary_keeps_quality_and_convergence_terms():
    summary = compute_relative_quality_summary(
        eval_rows=[
            {"step": 5000, "convergence_speed": 20.0, "sharpness_mean": 0.7, "lpips_mean": 0.2},
            {"step": 7000, "convergence_speed": 21.0, "sharpness_mean": 0.75, "lpips_mean": 0.18},
        ],
        baseline_eval_history=[
            {"step": 5000, "convergence_speed": 19.0, "sharpness_mean": 0.68, "lpips_mean": 0.25},
            {"step": 7000, "convergence_speed": 20.0, "sharpness_mean": 0.72, "lpips_mean": 0.22},
        ],
        loss_by_step={5000: 0.4, 7000: 0.35},
        elapsed_by_step={5000: 20.0, 7000: 30.0},
        baseline_loss_by_step_override={5000: 0.6, 7000: 0.5},
        t_eval_best=7000,
        t_end=7000,
        include_breakdown=True,
        score_reference_step=7000,
    )

    comparison = summary["baseline_comparison"]
    assert comparison["score_reference_step"] == 7000
    assert comparison["loss_at_7000_run"] == 0.35
    assert comparison["loss_at_7000_base"] == 0.5
    assert comparison["r_convergence"] == 0.15000000000000002
    assert "r_quality" in comparison
    assert summary["relative_score"] == comparison["r_quality"] + comparison["r_convergence"]

