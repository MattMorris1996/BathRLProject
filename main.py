from Agent import Agent
from AgentLogger import AgentLogger


def main():
    n_episodes = 250
    n_agents = 20

    seed = 16
    render_episodes = [0, 50, 100, 150, 200, 250]
    print("Random Seed: ", str(seed))

    logger = AgentLogger(
        _live_plot=False,  # Generate live plot
        _pickle_log=True,  # Save agent metrics to pickle log
        _saving_interval=25,  # Save to pickle log at interval
        _render_recording=True,  # Save agent rendering to gif
        _render_list=render_episodes,  # list of rendered episodes
        _console_log=True,  # log metrics to console
    )

    agents = [
        Agent(
            num_episodes=n_episodes,
            seed=seed,
            agent_num=nth_agent,
            logger=logger
        ) for nth_agent in range(n_agents)
    ]

    for agent in agents:
        agent.train()


if __name__ == "__main__":
    main()