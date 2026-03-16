"""Application entry point for the Monitoring RTSP GUI."""

import argparse
import logging
import traceback

from monitoring.app import main as run_app
from monitoring.runtime_helpers import app_log


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
    logging.getLogger(__name__).info("launcher startup")
    try:
        run_app(windowed=args.windowed)
    except Exception as exc:
        logging.getLogger(__name__).exception("launcher unhandled shutdown exception")
        app_log("error", "launcher unhandled shutdown exception", source="main", level="ERROR", details=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        logging.getLogger(__name__).info("launcher shutdown")


if __name__ == "__main__":
    main()
