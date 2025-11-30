import wandb
from agent import Agent

def main():
    # Start sweep run
    run = wandb.init()

    # THIS is the actual sweep config
    config = run.config

    # Create agent with config
    agent = Agent(config)

    agent.train(run)

    run.finish()


if __name__ == "__main__":
    main()
