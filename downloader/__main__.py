"""CLI entry point for Deribit data downloader.

Usage:
    PYTHONPATH=. python -m downloader
    PYTHONPATH=. python -m downloader --endpoints dvol --currencies BTC
    PYTHONPATH=. python -m downloader --endpoints trades --start 2023-05-01 --end 2023-06-01
    PYTHONPATH=. python -m downloader --help
"""

import argparse
import logging
import sys

from downloader.config import DownloadConfig
from downloader.runner import run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Quiet noisy libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Deribit historical data downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download everything (BTC+ETH, all endpoints)
  python -m downloader --client-id XXX --client-secret YYY

  # Just DVOL at 5-min resolution
  python -m downloader --endpoints dvol --resolution 5

  # BTC option trades only, specific dates
  python -m downloader --endpoints trades --currencies BTC --start 2023-06-01 --end 2023-07-01

  # Mark prices for ETH options
  python -m downloader --endpoints mark_price --currencies ETH

  # Unauthenticated (slower, ~10 req/s)
  python -m downloader --endpoints dvol historical_vol
""")

    parser.add_argument("--currencies", nargs="+", default=["BTC", "ETH"])
    parser.add_argument("--kinds", nargs="+", default=["option"])
    parser.add_argument("--start", default="2023-05-01")
    parser.add_argument("--end", default="2024-04-01")
    parser.add_argument("--endpoints", nargs="+",
                        default=["dvol", "historical_vol", "candles", "mark_price"],
                        choices=["trades", "dvol", "mark_price", "historical_vol", "candles"])
    parser.add_argument("--dvol-resolution", default="3600", help="DVOL resolution (3600=1h, 43200=12h, 1D)")
    parser.add_argument("--candle-resolution", default="5", help="Candle resolution in minutes")
    parser.add_argument("--output-dir", default="data/deribit")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--client-id", default="2kCTTPjV")
    parser.add_argument("--client-secret", default="gQJjU-OOImmA-kbCCnLqjm0wTtN24WdboLW6q7tQjfI")
    parser.add_argument("--candle-instruments", nargs="+",
                        default=["BTC-PERPETUAL", "ETH-PERPETUAL"])
    parser.add_argument("--trades-sample-days", type=int, default=1,
                        help="Download 1 day per N days (1=daily, 7=weekly)")
    parser.add_argument("--trades-max-pages", type=int, default=0,
                        help="Max pages per day (0=unlimited, 20=20K trades cap)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = DownloadConfig(
        currencies=args.currencies,
        kinds=args.kinds,
        start=args.start,
        end=args.end,
        endpoints=args.endpoints,
        dvol_resolution=args.dvol_resolution,
        candle_resolution=args.candle_resolution,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        client_id=args.client_id,
        client_secret=args.client_secret,
        candle_instruments=args.candle_instruments,
        trades_sample_days=args.trades_sample_days,
        trades_max_pages=args.trades_max_pages,
    )

    print(f"Deribit Downloader — {'authenticated' if config.authenticated else 'unauthenticated'}")
    print(f"  Currencies: {config.currencies}")
    print(f"  Endpoints:  {config.endpoints}")
    print(f"  Range:      {config.start} to {config.end}")
    print(f"  Output:     {config.output_dir}")
    print(f"  Workers:    {config.max_workers}")
    print()

    run(config)


if __name__ == "__main__":
    main()
