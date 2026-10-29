from src.experiment import ExperimentConfig, run_baseline_experiment


def test_baseline_smoke_run():
    result = run_baseline_experiment(ExperimentConfig(seed=0, num_nodes=12, num_features=5, num_classes=3, hidden_dim=4))
    assert 'accuracy' in result
    assert 'macro_f1' in result
    assert result['num_hyperboxes'] >= 1
