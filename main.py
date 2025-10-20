"""Application entry point for the Monitoring RTSP GUI."""

import argparse

from monitoring.app import main as run_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APP-MONITORING launcher")
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Run application in a window instead of full screen",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_app(windowed=args.windowed)


if __name__ == "__main__":
    main()
