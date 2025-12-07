import wandb
from types import SimpleNamespace
from agent import Agent


if __name__ == "__main__":

    config = SimpleNamespace(
        model_name="small",   # simple / resnet-pre-trained / vgg
        lr=1e-3,
        batch_size=128,
        epochs=50,
        image_size=128,
    )

    run = wandb.init(
        project="10-Animals",
        name="single_run_test",
        config=config.__dict__  # log config to W&B
    )

    agent = Agent(config)
    best_f1 = agent.train(run)

    print("\n========================")
    print(f"Finished single test run")
    print(f"Best F1 score: {best_f1:.4f}")
    print("========================\n")
