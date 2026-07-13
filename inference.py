import argparse

import torch
import torch.nn as nn

from train import _build_loaders, build_model, test_model
from utils import load_config


def run_inference(dataset, checkpoint_path, config, fd=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _ = _build_loaders(dataset, config, fd=fd)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim = ckpt.get("input_dim", test_loader.dataset.X.shape[-1])

    model_cfg = config["model"]
    regime_cfg = config.get("regime", {})
    model = build_model(input_dim, model_cfg, regime_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    criterion = nn.MSELoss()
    calibration = ckpt["calibration"]

    resolved_fd = fd
    if dataset.upper() == "CMAPSS" and resolved_fd is None:
        resolved_fd = config.get("data", {}).get("cmapss_fd", "FD003")

    return test_model(model, test_loader, criterion, device, calibration, dataset=dataset, fd=resolved_fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, required = True, choices = ["CBM", "CMAPSS"])
    parser.add_argument(
        "--fd", type=str, default = None, choices = ["FD001", "FD002", "FD003", "FD004"],
        help="C-MAPSS subset (CMAPSS only; overrides config.yaml's data.cmapss_fd).",
    )
    parser.add_argument("--checkpoint", type = str, required = True)
    parser.add_argument("--config", type = str, default = "./configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_inference(args.dataset, args.checkpoint, config, fd = args.fd)
