import argparse

from src.compare import compare
from src.config import load_config
from src.describe import describe
from src.download import download
from src.dynamics import track
from src.features import build_features
from src.prepare import prepare
from src.robustness import robustness
from src.synthetic import synthetic
from src.validate import validate


def main():
    parser = argparse.ArgumentParser(description="Dynamic attributed network of Russian municipalities")
    parser.add_argument("step", choices=["download", "prepare", "features", "compare", "track", "robustness", "describe", "validate", "synthetic"])
    parser.add_argument("--all", action="store_true", help="download every source, not only the contest package")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.step == "download":
        download(config, ["package", "mobility", "borders", "population", "labor", "prices"] if args.all else ["package", "mobility"])
    elif args.step == "prepare":
        prepare(config)
    elif args.step == "features":
        build_features(config)
    elif args.step == "compare":
        compare(config)
    elif args.step == "track":
        track(config)
    elif args.step == "robustness":
        robustness(config)
    elif args.step == "describe":
        describe(config)
    elif args.step == "validate":
        validate(config)
    else:
        synthetic(config)


if __name__ == "__main__":
    main()