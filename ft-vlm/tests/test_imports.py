def test_imports():
    import ft_vlm
    from ft_vlm.dataset.chartqa import load_chartqa_dataset
    from ft_vlm.model.qwen2_vl import QLoRAConfig, build_model_and_processor
    from ft_vlm.fine_tuning.train import TrainConfig
    from ft_vlm.dataset.local_json import load_local_json_dataset

    assert hasattr(ft_vlm, "__file__")
    assert QLoRAConfig is not None
    assert TrainConfig is not None
    assert load_local_json_dataset is not None
