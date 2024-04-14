import argparse


def build_args():
    parameters = [
        {"name": "--hdf", "type": str, "default": "",
         "help": "Path to the HDF file containing the latent vectors."},
    ]
    parser = argparse.ArgumentParser(description="Visualize Latent Space")
    for param in parameters:
        kwargs = {k: v for k, v in param.items() if k != "name"}
        parser.add_argument(param["name"], **kwargs)
    return parser.parse_args()


def shorten_middle(s: str, max_len: int, placeholder='...') -> str:
    if len(s) <= max_len:
        return s
    else:
        n_1 = (max_len - len(placeholder)) // 2
        n_2 = max_len - len(placeholder) - n_1
        return s[:n_1] + placeholder + s[-n_2:]
