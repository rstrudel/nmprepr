import os
import click

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import set_seed, setup_logger

from nmp.launcher.sac import sac
from nmp import settings


@click.command(help="nmp.train env_name exp_name")
@click.argument("env-name", type=str)
@click.argument("exp-dir", type=str)
@click.option("-s", "--seed", default=None, type=int)
@click.option("-resume", "--resume/--no-resume", is_flag=True, default=False)
@click.option("-mode", "--mode", default="her")
@click.option("-archi", "--archi", default="pointnet")
@click.option("-epochs", "--epochs", default=3000, type=int)
@click.option("-rscale", "--reward-scale", default=1, type=float)
@click.option("-h-dim", "--hidden-dim", default=256, type=int)
@click.option("-bs", "--batch-size", default=256, type=int)
@click.option("-lr", "--learning-rate", default=3e-4, type=float)
@click.option("-n-layers", "--n-layers", default=3, type=int)
@click.option("-tau", "--soft-target-tau", default=5e-3, type=float)
@click.option("-auto-alpha", "--auto-alpha/--no-auto-alpha", is_flag=True, default=True)
@click.option("-alpha", "--alpha", default=0.1, type=float)
@click.option("-frac-goal-replay", "--frac-goal-replay", default=0.8, type=float)
@click.option("-horizon", "--horizon", default=80, type=int)
@click.option("-rbs", "--replay-buffer-size", default=int(1e6), type=int)
@click.option("-cpu", "--cpu/--no-cpu", is_flag=True, default=False)
@click.option(
    "-snap-mode",
    "--snapshot-mode",
    default="last",
    type=str,
    help="all, last, gap, gap_and_last, none",
)
@click.option("-snap-gap", "--snapshot-gap", default=10, type=int)
def main(
    env_name,
    exp_dir,
    seed,
    resume,
    mode,
    archi,
    epochs,
    reward_scale,
    hidden_dim,
    batch_size,
    learning_rate,
    n_layers,
    soft_target_tau,
    auto_alpha,
    alpha,
    frac_goal_replay,
    horizon,
    replay_buffer_size,
    snapshot_mode,
    snapshot_gap,
    cpu,
):
    valid_modes = ["vanilla", "her"]
    valid_archi = [
        "mlp",
        "cnn",
        "pointnet",
    ]
    if mode not in valid_modes:
        raise ValueError(f"Unknown mode: {mode}")
    if archi not in valid_archi:
        raise ValueError(f"Unknown network archi: {archi}")

    machine_log_dir = settings.log_dir()
    exp_dir = os.path.join(machine_log_dir, exp_dir, f"seed{seed}")
    # multi-gpu and batch size scaling
    replay_buffer_size = replay_buffer_size
    num_expl_steps_per_train_loop = 1000
    num_eval_steps_per_epoch = 1000
    min_num_steps_before_training = 1000
    num_trains_per_train_loop = 1000
    # learning rate and soft update linear scaling
    policy_lr = learning_rate
    qf_lr = learning_rate
    variant = dict(
        env_name=env_name,
        algorithm="sac",
        version="normal",
        seed=seed,
        resume=resume,
        mode=mode,
        archi=archi,
        replay_buffer_kwargs=dict(max_replay_buffer_size=replay_buffer_size,),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            num_epochs=epochs,
            num_eval_steps_per_epoch=num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop=num_expl_steps_per_train_loop,
            num_trains_per_train_loop=num_trains_per_train_loop,
            min_num_steps_before_training=min_num_steps_before_training,
            max_path_length=horizon,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=soft_target_tau,
            target_update_period=1,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            reward_scale=reward_scale,
            use_automatic_entropy_tuning=auto_alpha,
            alpha=alpha,
        ),
        qf_kwargs=dict(hidden_dim=hidden_dim, n_layers=n_layers),
        policy_kwargs=dict(hidden_dim=hidden_dim, n_layers=n_layers),
        log_dir=exp_dir,
    )
    if mode == "her":
        variant["replay_buffer_kwargs"].update(
            dict(
                fraction_goals_rollout_goals=1
                - frac_goal_replay,  # equal to k = 4 in HER paper
                fraction_goals_env_goals=0,
            )
        )
    set_seed(seed)

    setup_logger_kwargs = {
        "exp_prefix": exp_dir,
        "variant": variant,
        "log_dir": exp_dir,
        "snapshot_mode": snapshot_mode,
        "snapshot_gap": snapshot_gap,
    }
    setup_logger(**setup_logger_kwargs)
    ptu.set_gpu_mode(not cpu, distributed_mode=False)
    print(f"Start training...")
    sac(variant)


if __name__ == "__main__":
    main()
