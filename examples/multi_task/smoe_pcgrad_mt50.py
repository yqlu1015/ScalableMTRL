from dataclasses import dataclass
from pathlib import Path

import tyro

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from metaworld_algorithms.config.nn import SparseMoEConfig
from metaworld_algorithms.config.optim import OptimizerConfig, PCGradConfig
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
    num_experts: int = 8
    num_shared_experts: int = 2
    k_active_experts: int = 4


def main() -> None:
    args = tyro.cli(Args)

    run = Run(
        run_name=f"mt50_smoe_pcgrad_e{args.num_experts}_se{args.num_shared_experts}_k{args.k_active_experts}",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            env_id="MT50",
            terminate_on_success=False,
        ),
        algorithm=MTSACConfig(
            num_tasks=50,
            gamma=0.99,
            actor_config=ContinuousActionPolicyConfig(
                network_config=SparseMoEConfig(
                    num_tasks=50, optimizer=PCGradConfig(num_tasks=50, max_grad_norm=1.0),
                    num_experts=args.num_experts,
                    num_shared_experts=args.num_shared_experts,
                    k_active_experts=args.k_active_experts,
                ),
                log_std_min=-10,
                log_std_max=2,
            ),
            critic_config=QValueFunctionConfig(
                network_config=SparseMoEConfig(
                    num_tasks=50,
                    optimizer=PCGradConfig(num_tasks=50, max_grad_norm=1.0),
                    num_experts=args.num_experts,
                    num_shared_experts=args.num_shared_experts,
                    k_active_experts=args.k_active_experts,
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
        ),
        checkpoint=True,
        resume=args.resume,
    )
    print("Run initialized")

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