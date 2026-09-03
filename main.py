from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anyio
import httpx
import polars as pl
import psutil
import tooltime as tt
import typer
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv(override=True)
SITEMAP_INDEX_URL = "https://assets.grokipedia.com/sitemap/sitemap-index.xml"
HF_REPO_ID = "caentzminger/grokipedia-urls"
CONCURRENCY_LIMIT = 20

pl.Config.set_engine_affinity("streaming")

app = typer.Typer(help="Grokipedia URL dataset collector")


def fmt_num(n: int | float) -> str:
    """Format a number with thousands separators.

    Args:
        n: Number to format (int or float).

    Returns:
        String with commas as thousands separators.
    """
    return f"{n:,}"


def fmt_time(seconds: float) -> str:
    """Format seconds into a human-readable time string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Readable time string (e.g., "1m 30s" or "01:30:00").
    """
    if seconds < 60:
        return f"{tt.timelength_seconds_to_phrase(seconds)}"
    return tt.timelength_seconds_to_clock_phrase(seconds)


def fmt_mem(bytes_val: int) -> str:
    """Format bytes into a human-readable memory string.

    Args:
        bytes_val: Memory in bytes.

    Returns:
        Formatted string in MB (if < 1GB) or GB.
    """
    mb = bytes_val / 1024 / 1024
    if mb < 1000:
        return f"{mb:.1f} MB"
    return f"{mb / 1024:.2f} GB"


def needs_push(new_data_path: Path, repo_id: str) -> bool:
    """Compare local dataset with HuggingFace remote to determine if push is needed.

    Compares both row count and content hash (via URL hash) between local and remote.

    Args:
        new_data_path: Path to local parquet file.
        repo_id: HuggingFace dataset repo ID (e.g., "user/dataset").

    Returns:
        True if push is needed (different content or remote unavailable), False otherwise.
    """
    try:
        token = os.getenv("HF_TOKEN")
        storage_opts = {"token": token} if token else {}

        existing = pl.scan_parquet(
            f"hf://datasets/{repo_id}/data/*.parquet", storage_options=storage_opts
        )
        new = pl.scan_parquet(new_data_path)

        existing_df = existing.select(pl.len()).collect()
        new_df = new.select(pl.len()).collect()
        existing_count: int = existing_df.item(0, 0)  # type: ignore[unresolved-attribute]
        new_count: int = new_df.item(0, 0)  # type: ignore[unresolved-attribute]

        logger.log(
            f"HF count: {fmt_num(existing_count)}, Local count: {fmt_num(new_count)}",
            phase="DEBUG",
        )

        if existing_count != new_count:
            logger.log(
                f"Row count differs: {fmt_num(existing_count)} vs {fmt_num(new_count)}"
            )
            return True

        existing_hash_df = existing.select(pl.col("url").hash().sum()).collect()
        new_hash_df = new.select(pl.col("url").hash().sum()).collect()
        existing_hash = existing_hash_df.item(0, 0)  # type: ignore[unresolved-attribute]
        new_hash = new_hash_df.item(0, 0)  # type: ignore[unresolved-attribute]

        logger.log(f"HF hash: {existing_hash}, Local hash: {new_hash}", phase="DEBUG")

        if existing_hash != new_hash:
            logger.log("Content hash differs")
            return True

        return False
    except Exception as e:
        logger.log(f"Could not compare with remote: {e}", phase="WARN")
        return True


class Logger:
    """Structured logger for tracking execution phases and resource usage.

    Attributes:
        quiet: If True, only prints message to stderr without timestamp/phase.
    """

    def __init__(self, quiet: bool = False):
        """Initialize the logger.

        Args:
            quiet: If True, suppresses timestamp and phase prefix (default: False).
        """
        self.quiet = quiet

    def log(self, message: str, phase: str = "INFO"):
        """Log a message with optional phase.

        Args:
            message: The message to log.
            phase: Log level/phase (default: "INFO").
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{timestamp}] {phase}: {message}"
        if self.quiet:
            print(message, file=sys.stderr)
        else:
            print(line)

    def phase_start(self, name: str):
        """Log the start of a processing phase.

        Args:
            name: Name of the phase being started.
        """
        self.log(f"Starting: {name}", phase="PHASE")

    def phase_end(
        self,
        name: str,
        duration: float,
        memory_delta: int,
        peak_rss: int,
        cpu_user: float,
    ):
        """Log the completion of a processing phase with metrics.

        Args:
            name: Name of the phase that completed.
            duration: Total elapsed time in seconds.
            memory_delta: Change in memory usage in bytes.
            peak_rss: Peak resident set size in bytes.
            cpu_user: CPU user time in seconds.
        """
        self.log(
            f"Completed: {name} | Duration: {fmt_time(duration)} | "
            f"Memory delta: {fmt_mem(memory_delta)} | Peak RSS: {fmt_mem(peak_rss)} | "
            f"CPU user: {fmt_time(cpu_user)}",
            phase="PHASE",
        )

    def summary(
        self, total_urls: int, peak_rss: int, output_path: Path, temp_dir: Path
    ):
        """Log final summary of the collection run.

        Args:
            total_urls: Total number of URLs collected.
            peak_rss: Peak memory usage in bytes.
            output_path: Path to the output parquet file.
            temp_dir: Path to the temporary directory used for chunks.
        """
        self.log(f"Dataset saved to: {output_path}", phase="DONE")
        self.log(f"Total URLs: {fmt_num(total_urls)}", phase="DONE")
        self.log(f"Peak memory: {fmt_mem(peak_rss)}", phase="DONE")
        self.log(f"Temp chunks: {temp_dir}", phase="DEBUG")


logger: Logger = Logger()


@contextmanager
def track_resources(phase_name: str):
    """Context manager for tracking resource usage during a processing phase.

    Tracks execution time, memory delta, peak RSS, and CPU user time.
    Logs phase start and end with all metrics via the Logger.

    Args:
        name: Name of the phase being tracked.
    """
    process = psutil.Process()
    start_time = time.perf_counter()
    start_mem = process.memory_info().rss
    start_cpu = process.cpu_times()

    logger.phase_start(phase_name)

    yield

    end_time = time.perf_counter()
    end_mem = process.memory_info().rss
    end_cpu = process.cpu_times()
    duration = end_time - start_time
    memory_delta = end_mem - start_mem
    peak_rss = end_mem
    cpu_user = end_cpu.user - start_cpu.user

    logger.phase_end(phase_name, duration, memory_delta, peak_rss, cpu_user)


async def fetch_sitemap_index(client: httpx.AsyncClient) -> list[str]:
    """Fetch and parse the sitemap index XML to get all child sitemap URLs.

    Args:
        client: HTTPX async client for making requests.

    Returns:
        List of sitemap URLs found in the index.

    Raises:
        httpx.HTTPStatusError: If the request fails.
    """
    response = await client.get(SITEMAP_INDEX_URL)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    sitemap_locs = root.findall(".//sm:loc", ns)
    if not sitemap_locs:
        sitemap_locs = root.findall(".//loc")
    return [loc.text for loc in sitemap_locs if loc.text]


async def fetch_sitemap_to_records(client: httpx.AsyncClient, url: str) -> list[dict]:
    """Fetch a sitemap XML and extract all URL records as dictionaries.

    Extracts URL, lastmod, changefreq, priority, and sitemap_source fields.
    Returns empty list on error (logs warning).

    Args:
        client: HTTPX async client for making requests.
        url: URL of the sitemap to fetch.

    Returns:
        List of dictionaries containing URL record data.
    """
    try:
        response = await client.get(url)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        url_elements = root.findall(".//sm:url", ns)
        if not url_elements:
            url_elements = root.findall(".//url")

        results = []
        for url_elem in url_elements:
            loc = url_elem.find("sm:loc", ns)
            if loc is None:
                loc = url_elem.find("loc")
            if loc is None or not loc.text:
                continue

            lastmod = url_elem.find("sm:lastmod", ns)
            if lastmod is None:
                lastmod = url_elem.find("lastmod")

            changefreq = url_elem.find("sm:changefreq", ns)
            if changefreq is None:
                changefreq = url_elem.find("changefreq")

            priority = url_elem.find("sm:priority", ns)
            if priority is None:
                priority = url_elem.find("priority")

            results.append(
                {
                    "url": loc.text,
                    "lastmod": lastmod.text if lastmod is not None else None,
                    "changefreq": changefreq.text if changefreq is not None else None,
                    "priority": float(priority.text)
                    if priority is not None and priority.text
                    else None,
                    "sitemap_source": url,
                }
            )

        return results
    except Exception as e:
        logger.log(f"Error fetching {url}: {e}", phase="ERROR")
        return []


async def fetch_and_stream_to_chunks(
    sitemap_urls: list[str], temp_dir: Path, concurrency: int
) -> int:
    """Concurrently fetch all sitemaps and stream results to chunked parquet files.

    Fetches sitemaps concurrently using anyio, accumulating records and writing
    to parquet chunks when the target size (25k records) is reached. Each record
    gets a "fetched_at" timestamp.

    Args:
        sitemap_urls: List of sitemap URLs to fetch.
        temp_dir: Directory to write chunk parquet files.
        concurrency: Maximum number of concurrent requests.

    Returns:
        Number of chunks created.
    """
    fetched_at = datetime.now(timezone.utc).isoformat()
    chunk_index = 0
    current_records = []
    records_in_current_chunk = 0
    target_records_per_chunk = 25000
    total_records = 0

    limits = httpx.Limits(max_connections=concurrency)

    async with httpx.AsyncClient(limits=limits, timeout=60.0) as client:
        semaphore = anyio.Semaphore(concurrency)

        async def fetch_one(url: str) -> list[dict]:
            async with semaphore:
                return await fetch_sitemap_to_records(client, url)

        async def worker(url: str) -> None:
            nonlocal \
                current_records, \
                records_in_current_chunk, \
                chunk_index, \
                total_records

            records = await fetch_one(url)
            for r in records:
                r["fetched_at"] = fetched_at
            current_records.extend(records)
            total_records += len(records)
            records_in_current_chunk += len(records)

            if records_in_current_chunk >= target_records_per_chunk:
                df = pl.from_records(current_records)
                chunk_path = temp_dir / f"chunk_{chunk_index:04d}.parquet"
                df.write_parquet(chunk_path)
                logger.log(
                    f"Wrote chunk {chunk_index}: {fmt_num(len(df))} rows", phase="CHUNK"
                )
                chunk_index += 1
                current_records = []
                records_in_current_chunk = 0

        async with anyio.create_task_group() as tg:
            for url in sitemap_urls:
                tg.start_soon(worker, url)

    if current_records:
        df = pl.from_records(current_records)
        chunk_path = temp_dir / f"chunk_{chunk_index:04d}.parquet"
        df.write_parquet(chunk_path)
        logger.log(
            f"Wrote final chunk {chunk_index}: {fmt_num(len(df))} rows ({fmt_num(total_records)} total)",
            phase="CHUNK",
        )
        chunk_index += 1

    return chunk_index


@app.command()
def main(
    output: Path = Path("data/grokipedia_urls.parquet"),
    push: bool = False,
    hf_repo: Optional[str] = None,
    concurrency: int = CONCURRENCY_LIMIT,
    sort: bool = True,
    quiet: bool = False,
    force_push: bool = False,
    keep_temp: bool = False,
):
    """Collect URLs from Grokipedia sitemap and optionally push to HuggingFace.

    This command performs three phases:
    1. Fetch the sitemap index to get all child sitemap URLs
    2. Concurrently fetch all sitemaps and stream to chunked parquet files
    3. Combine chunks into a single sorted parquet file

    If --push is specified, the dataset is pushed to HuggingFace Hub.
    Uses HF_TOKEN env var for authentication.

    Args:
        output: Path for output parquet file (default: data/grokipedia_urls.parquet).
        push: Whether to push the dataset to HuggingFace Hub.
        hf_repo: Custom HuggingFace repo ID (default: uses HF_REPO_ID).
        concurrency: Max concurrent HTTP requests (default: 20).
        sort: Whether to sort output by URL (default: True).
        quiet: Suppress timestamps/phase info, show only messages.
        force_push: Skip change detection and force push to HF.
        keep_temp: Keep temporary chunk files after completion.
    """
    global logger
    logger = Logger(quiet=quiet)

    output.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="grokipedia_chunks_"))
    logger.log(f"Temp directory: {temp_dir}", phase="INFO")

    with track_resources("Phase 1: Fetch sitemap index"):
        sitemap_urls = anyio.run(fetch_sitemap_index, httpx.AsyncClient())
        logger.log(f"Found {fmt_num(len(sitemap_urls))} child sitemaps", phase="INFO")

    with track_resources("Phase 2: Fetch and chunk sitemaps"):
        num_chunks = anyio.run(
            fetch_and_stream_to_chunks, sitemap_urls, temp_dir, concurrency
        )
        logger.log(f"Created {fmt_num(num_chunks)} chunk files", phase="INFO")

    with track_resources("Phase 3: Lazy combine and write"):
        changefreq_categories = [
            "always",
            "hourly",
            "daily",
            "weekly",
            "monthly",
            "yearly",
            "never",
        ]

        lf = pl.scan_parquet(temp_dir / "chunk_*.parquet")

        lf = lf.with_columns(
            [
                pl.col("lastmod").str.to_date(format="%Y-%m-%d", strict=False),
                pl.col("changefreq").cast(pl.Enum(categories=changefreq_categories)),
                pl.col("priority").cast(pl.Float32),
                pl.col("fetched_at").str.to_datetime(
                    format="%Y-%m-%dT%H:%M:%S%.f%z", strict=False
                ),
            ]
        )

        if sort:
            lf = lf.sort("url")

        lf.sink_parquet(output, row_group_size=100_000)

    process = psutil.Process()
    end_mem = process.memory_info().rss / 1024 / 1024

    df = pl.read_parquet(output)

    logger.summary(len(df), int(end_mem * 1024 * 1024), output, temp_dir)

    if not keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.log("Cleaned up temp directory", phase="DEBUG")

    if push:
        token = os.getenv("HF_TOKEN")
        if not token:
            logger.log("HF_TOKEN not found in environment", phase="ERROR")
            raise typer.Exit(code=1)

        repo_id = hf_repo or HF_REPO_ID

        if not force_push:
            logger.log("Checking for updates...", phase="INFO")
            if not needs_push(output, repo_id):
                logger.log(
                    "Dataset unchanged, skipping push. Use --force-push to override.",
                    phase="INFO",
                )
                raise typer.Exit(code=0)

        hf_dataset = Dataset.from_parquet(str(output))
        hf_dataset.push_to_hub(repo_id, token=token, private=True)

        logger.log(f"Pushed to https://huggingface.co/datasets/{repo_id}", phase="DONE")


if __name__ == "__main__":
    app()
