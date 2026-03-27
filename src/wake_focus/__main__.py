"""
Wake Focus - Entry Point

Usage:
    python -m wake_focus [--config CONFIG_PATH] [--edge]

Or via installed script:
    wake-focus [--config CONFIG_PATH] [--edge]
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Wake Focus — Driver Monitoring & Fleet Navigation System",
        prog="wake-focus",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--edge",
        action="store_true",
        help="Enable edge mode (optimized for Orange Pi Zero 2W / ARM64)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level",
    )

    args = parser.parse_args()

    # Import here to avoid slow imports before arg parsing
    from wake_focus.config import Config
    from wake_focus.app import WakeFocusApp

    config = Config(config_path=args.config, edge_mode=args.edge)

    if args.log_level:
        config._data.setdefault("app", {})["log_level"] = args.log_level

    app = WakeFocusApp(config)
    sys.exit(app.run())


if __name__ == "__main__":
    main()
