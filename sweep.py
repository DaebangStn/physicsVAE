import sys
import wandb
from run import main


def main_with_args():
    orig_argv = sys.argv
    sys.argv = [
        'run.py',
        '--cfg_train', 'configs/train/styleAlgo.yaml',
        '--cfg_env', 'configs/env/keypointMaxObsTask.yaml',
        '--wandb_proj', 'ase'
    ]
    main()
    sys.argv = orig_argv


wandb.login()
sweep_config = {
    "name": "task_reward_scale",
    "method": "grid",
    "parameters": {
        "max_frames": {
            "values": [400000000]
        },
        "task_scale": {
            "values": [5, 10, 20, 40]
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="ase")
wandb.agent(sweep_id, function=main_with_args)
