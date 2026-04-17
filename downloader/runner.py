"""Parallel download orchestrator."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from downloader.client import DeribitClient
from downloader.config import DownloadConfig
from downloader.tasks.trades import download_day, build_work_queue
from downloader.tasks.dvol import download_dvol
from downloader.tasks.historical_vol import download_historical_vol
from downloader.tasks.mark_price import download_mark_prices
from downloader.tasks.candles import download_candles

logger = logging.getLogger(__name__)


def run(config: DownloadConfig) -> dict:
    """Run all configured downloads. Returns summary stats."""
    client = DeribitClient(config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats = {}

    for endpoint in config.endpoints:
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("  Downloading: %s", endpoint.upper())
        logger.info("=" * 60)

        if endpoint == "historical_vol":
            for currency in config.currencies:
                download_historical_vol(client, currency, output)
            stats[endpoint] = {"runtime": time.time() - t0}

        elif endpoint == "dvol":
            for currency in config.currencies:
                download_dvol(client, currency, config.start, config.end,
                              config.dvol_resolution, output)
            stats[endpoint] = {"runtime": time.time() - t0}

        elif endpoint == "candles":
            for instrument in config.candle_instruments:
                download_candles(client, instrument, config.start, config.end,
                                 config.candle_resolution, output)
            stats[endpoint] = {"runtime": time.time() - t0}

        elif endpoint == "mark_price":
            total = 0
            for currency in config.currencies:
                n = download_mark_prices(client, currency, config.start,
                                         config.end, output)
                total += n
            stats[endpoint] = {"instruments": total, "runtime": time.time() - t0}

        elif endpoint == "trades":
            work = build_work_queue(config.currencies, config.kinds,
                                    config.start, config.end,
                                    sample_every_n_days=config.trades_sample_days)
            logger.info("  %d sampled days to download (every %d days, max %d pages/day)",
                        len(work), config.trades_sample_days, config.trades_max_pages)

            completed = 0
            failed = 0
            skipped = 0
            total_trades = 0

            with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
                futures = {
                    pool.submit(download_day, client, c, k, d, output,
                                max_pages=config.trades_max_pages): (c, k, d)
                    for c, k, d in work
                }
                for fut in as_completed(futures):
                    c, k, d = futures[fut]
                    try:
                        result = fut.result()
                        if result is None:
                            skipped += 1  # empty day
                        else:
                            completed += 1
                    except Exception as ex:
                        failed += 1
                        logger.error("  FAILED %s %s %s: %s", c, k,
                                     d.strftime("%Y-%m-%d"), ex)

                    done = completed + failed + skipped
                    if done % 10 == 0 or done == len(work):
                        elapsed = time.time() - t0
                        rate = done / elapsed if elapsed > 0 else 0
                        remaining = len(work) - done
                        eta_s = remaining / rate if rate > 0 else 0
                        eta_h = eta_s / 3600
                        logger.info("  Progress: %d/%d (ok=%d, fail=%d, empty=%d) "
                                    "%.2f days/s  ETA %.1fh  %s",
                                    done, len(work), completed, failed, skipped,
                                    rate, eta_h, client.limiter.stats)

            stats[endpoint] = {
                "completed": completed, "failed": failed, "skipped": skipped,
                "runtime": time.time() - t0,
            }

        elapsed = time.time() - t0
        logger.info("  %s done in %.0fs", endpoint, elapsed)

    # Final summary
    logger.info("=" * 60)
    logger.info("  DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    for ep, s in stats.items():
        logger.info("  %-20s %s", ep, s)
    logger.info("  Rate limiter: %s", client.limiter.stats)

    return stats
