from dataclasses import dataclass
from pathlib import Path

import tyro

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from metaworld_algorithms.config.nn import VanillaNetworkConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import OffPolicyTrainingConfig
from metaworld_algorithms.envs import MetaworldConfig
from metaworld_algorithms.rl.algorithms import MTSACConfig
from metaworld_algorithms.run import Run


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = "MT10"
    wandb_entity: str | None = None
    data_dir: Path = Path("./run_results")
    resume: bool = False


def main() -> None:
    args = tyro.cli(Args)

    run = Run(
        run_name="mt10_mtsac",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            env_id="MT10",
            terminate_on_success=False,
            # max_episode_steps=150,  # keep the same as MOORE paper
            # evaluation_num_episodes=10,  # keep the same as MOORE paper
        ),
        algorithm=MTSACConfig(
            num_tasks=10,
            gamma=0.99,
            actor_config=ContinuousActionPolicyConfig(
                network_config=VanillaNetworkConfig(
                    optimizer=OptimizerConfig(max_grad_norm=1.0)
                ),
                log_std_max=2.0,  # keep the same as MOORE paper
                log_std_min=-20.0,  # keep the same as MOORE paper
            ),
            critic_config=QValueFunctionConfig(
                network_config=VanillaNetworkConfig(
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                )
            ),
            # temperature_optimizer_config=OptimizerConfig(lr=1e-4),  # keep the same as MOORE paper
            tau=0.005,  # keep the same as MOORE paper
            num_critics=2,
        ),
        training_config=OffPolicyTrainingConfig(
            total_steps=int(2e6),  # 20 epochs x 1e5 steps per epoch, keep the same as MOORE paper
            warmstart_steps=1500,  # keep the same as MOORE paper
            # evaluation_frequency=int(200_000 // 150),  # 2e6 total steps / 10 environments / 150 horizon, keep the same as MOORE paper
            buffer_size=int(1e6),  # keep the same as MOORE paper
            batch_size=1280,  # 128 x 10 environments, keep the same as MOORE paper
        ),
        checkpoint=True,
        resume=args.resume,
    )

    if args.track:
        # assert args.wandb_project is not None and args.wandb_entity is not None
        run.enable_wandb(
            project=args.wandb_project,
            # entity=args.wandb_entity,
            config=run,
            resume="allow",
        )

    run.start()


if __name__ == "__main__":
    main()
