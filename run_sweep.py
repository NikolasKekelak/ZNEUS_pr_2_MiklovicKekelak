import wandb
from agent import Agent

def main():

    run = wandb.init()
    config = run.config
    agent = Agent(config)
    agent.train(run)
    run.finish()


if __name__ == "__main__":
    main()
