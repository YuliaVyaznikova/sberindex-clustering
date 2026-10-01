import argparse

from src.compare import compare
from src.config import load_config
from src.download import download
from src.features import build_features
from src.prepare import prepare


def main():
    parser = argparse.ArgumentParser(description="Dynamic attributed network of Russian municipalities")
    parser.add_argument("step", choices=["download", "prepare", "features", "compare"])
    parser.add_argument("--all", action="store_true", help="download every source, not only the contest package")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.step == "download":
        download(config, ["package", "borders", "population", "labor", "prices"] if args.all else ["package"])
    elif args.step == "prepare":
        prepare(config)
    elif args.step == "features":
        build_features(config)
    else:
        compare(config)


if __name__ == "__main__":
    main()