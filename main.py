import argparse
from train import train_tcn, test_model
from utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = "CBM", choices = ["CBM", "CMAPSS"])
    parser.add_argument(
        "--fd", type = str, default = None, choices = ["FD001", "FD002", "FD003", "FD004"],
        help = "C-MAPSS subset (CMAPSS only; overrides config.yaml's data.cmapss_fd).",
    )
    parser.add_argument("--config", type = str, default = "./config.yaml")
    args = parser.parse_args()

    model, test_loader, criterion, device, calibration = train_tcn(args.config, args.dataset, fd=args.fd)

    resolved_fd = args.fd
    if args.dataset.upper() == "CMAPSS" and resolved_fd is None:
        resolved_fd = load_config(args.config).get("data", {}).get("cmapss_fd", "FD003")

    test_model(model, test_loader, criterion, device, calibration, dataset=args.dataset, fd = resolved_fd)


if __name__ == "__main__":
    main()


