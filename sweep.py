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
    "name": "test_sweep",
    "method": "grid",
    "parameters": {
        "max_frames": {
            "values": [100000, 200000, 300000]
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="ase")
wandb.agent(sweep_id, function=main_with_args)
