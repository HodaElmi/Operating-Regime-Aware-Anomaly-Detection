import argparse
import yaml

import torch
import torch.nn as nn

from data_pipeline import build_cbm_dataloaders, build_cmapss_dataloaders
from model import RegimeAwareTCN
from train import test_model


def _build_test_loader(dataset, config):
    data_cfg = config["data"]
    train_cfg = config["training"]
    regime_cfg = config.get("regime", {})

    if dataset.upper() == "CBM":
        _, _, test_loader, _ = build_cbm_dataloaders(
            path=data_cfg["path"],
            window_size=train_cfg["window_size"],
            stride=train_cfg["stride"],
            batch_size=train_cfg["batch_size"],
            split_train=data_cfg.get("split_train", 0.7),
            split_val=data_cfg.get("split_val", 0.15),
            split_test=data_cfg.get("split_test", 0.15),
            n_regimes=regime_cfg.get("n_regimes", 3),
            regime_feature_idx=regime_cfg.get("feature_idx", None),
        )
        return test_loader

    if dataset.upper() == "CMAPSS":
        _, _, test_loader, _ = build_cmapss_dataloaders(
            train_path=data_cfg.get("cmapss_train_path", "./CMAPSSData/train_1.npy"),
            test_path=data_cfg.get("cmapss_test_path", "./CMAPSSData/test_1.npy"),
            window_size=train_cfg["window_size"],
            stride=train_cfg["stride"],
            batch_size=train_cfg["batch_size"],
            val_split=data_cfg.get("val_split", 0.2),
            n_regimes=regime_cfg.get("n_regimes", 3),
            regime_feature_idx=regime_cfg.get("feature_idx", None),
            normal_rul=regime_cfg.get("cmapss_normal_rul", 100),
            anomaly_rul=regime_cfg.get("cmapss_anomaly_rul", 30),
        )
        return test_loader

    raise ValueError(f"Unknown dataset: {dataset}")


def run_inference(dataset, checkpoint_path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = _build_test_loader(dataset, config)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim = ckpt.get("input_dim", test_loader.dataset.X.shape[-1])

    model_cfg = config["model"]
    regime_cfg = config.get("regime", {})
    model = RegimeAwareTCN(
        input_dim=input_dim,
        channels=model_cfg["channels"],
        kernel_size=model_cfg["kernel_size"],
        dropout=model_cfg["dropout"],
        n_regimes=regime_cfg.get("n_regimes", 3),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    criterion = nn.MSELoss()
    calibration = ckpt["calibration"]

    return test_model(model, test_loader, criterion, device, calibration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["CBM", "CMAPSS"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_inference(args.dataset, args.checkpoint, config)
