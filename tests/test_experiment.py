from src.experiment import ExperimentConfig, run_baseline_experiment, run_refinement_ablation, run_softmax_experiment


def test_baseline_smoke_run():
    result = run_baseline_experiment(ExperimentConfig(seed=0, num_nodes=12, num_features=5, num_classes=3, hidden_dim=4))
    assert 'accuracy' in result
    assert 'macro_f1' in result
    assert result['num_hyperboxes'] >= 1


def test_softmax_and_refinement_modes_are_distinct():
    config = ExperimentConfig(seed=1, num_nodes=12, num_features=5, num_classes=3, hidden_dim=4, max_epochs=2, patience=1)
    softmax = run_softmax_experiment(config)
    geometry = run_refinement_ablation(config, mode='geometry_merge')
    graph = run_refinement_ablation(config, mode='graph_merge')
    assert softmax['num_hyperboxes'] == 0
    assert geometry['variants'][0]['name'] == 'geometry_merge'
    assert graph['variants'][0]['name'] == 'graph_merge'
