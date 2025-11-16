import wandb

run = wandb.init(
    project="10-Animals",
    name="baseline_cnn",
    config={
        "name" : "test_raw_cnn_pre_gen"
    }
)
