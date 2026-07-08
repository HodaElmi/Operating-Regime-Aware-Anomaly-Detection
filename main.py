import argparse
from train import train_tcn, test_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CBM", choices=["CBM", "CMAPSS"])
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()

    model, test_loader, criterion, device, calibration = train_tcn(args.config, args.dataset)
    test_model(model, test_loader, criterion, device, calibration)


if __name__ == "__main__":
    main()
