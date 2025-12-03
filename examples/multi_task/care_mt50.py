from dataclasses import dataclass
from pathlib import Path

import tyro

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from metaworld_algorithms.config.nn import CAREConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import OffPolicyTrainingConfig
from metaworld_algorithms.envs import MetaworldConfig
from metaworld_algorithms.rl.algorithms import MTSACConfig
from metaworld_algorithms.run import Run


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = True
    wandb_project: str | None = "MT50"
    wandb_entity: str | None = None
    data_dir: Path = Path("./run_results")
    resume: bool = False
    num_experts: int = 6


def main() -> None:
    args = tyro.cli(Args)

    run = Run(
        run_name="mt50_care",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            env_id="MT50",
            terminate_on_success=False,
            # evaluation_num_episodes=10,
        ),
        algorithm=MTSACConfig(
            num_tasks=50,
            gamma=0.99,
            actor_config=ContinuousActionPolicyConfig(
                network_config=CAREConfig(
                    num_tasks=50, 
                    optimizer=OptimizerConfig(lr=3e-4, max_grad_norm=1.0),
                    num_experts=args.num_experts,
                ),
                log_std_min=-10,
                log_std_max=2,
            ),
            critic_config=QValueFunctionConfig(
                network_config=CAREConfig(
                    num_tasks=50,
                    optimizer=OptimizerConfig(lr=3e-4, max_grad_norm=1.0),
                    num_experts=args.num_experts,
                )
            ),
            temperature_optimizer_config=OptimizerConfig(lr=1e-4),
            num_critics=2,
        ),
        training_config=OffPolicyTrainingConfig(
            total_steps=int(2e7),
            # warmstart_steps=1500,
            buffer_size=int(1e6),
            batch_size=128 * 50,
            # evaluation_frequency=100,
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
