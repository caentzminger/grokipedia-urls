from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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
    return f"{n:,}"


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{tt.timelength_seconds_to_phrase(seconds)}"
    return tt.timelength_seconds_to_clock_phrase(seconds)


def fmt_mem(bytes_val: int) -> str:
    mb = bytes_val / 1024 / 1024
    if mb < 1000:
        return f"{mb:.1f} MB"
    return f"{mb / 1024:.2f} GB"


def needs_push(new_data_path: Path, repo_id: str) -> bool:
    try:
        token = os.getenv("HF_TOKEN")
        storage_opts = {"token": token} if token else {}

        existing = pl.scan_parquet(
            f"hf://datasets/{repo_id}/data/*.parquet", storage_options=storage_opts
        )
        new = pl.scan_parquet(new_data_path)

        stats = existing.select(
            pl.len().alias("count"), pl.col("url").hash().sum().alias("hash")
        ).collect()
        existing_count: int = stats.item(0, "count")
        existing_hash: int = stats.item(0, "hash")

        new_stats = new.select(
            pl.len().alias("count"), pl.col("url").hash().sum().alias("hash")
        ).collect()
        new_count: int = new_stats.item(0, "count")
        new_hash: int = new_stats.item(0, "hash")

        logger.log(
            f"HF count: {fmt_num(existing_count)}, Local count: {fmt_num(new_count)}",
            phase="DEBUG",
        )
        logger.log(f"HF hash: {existing_hash}, Local hash: {new_hash}", phase="DEBUG")

        if existing_count != new_count:
            logger.log(
                f"Row count differs: {fmt_num(existing_count)} vs {fmt_num(new_count)}"
            )
            return True

        if existing_hash != new_hash:
            logger.log("Content hash differs")
            return True

        return False
    except Exception as e:
        logger.log(f"Could not compare with remote: {e}", phase="WARN")
        return True


class Logger:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet

    def log(self, message: str, phase: str = "INFO"):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{timestamp}] {phase}: {message}"
        if self.quiet:
            print(message, file=sys.stderr)
        else:
            print(line)

    def phase_start(self, name: str):
        self.log(f"Starting: {name}", phase="PHASE")

    def phase_end(
        self,
        name: str,
        duration: float,
        memory_delta: int,
        peak_rss: int,
        cpu_user: float,
    ):
        self.log(
            f"Completed: {name} | Duration: {fmt_time(duration)} | "
            f"Memory delta: {fmt_mem(memory_delta)} | Peak RSS: {fmt_mem(peak_rss)} | "
            f"CPU user: {fmt_time(cpu_user)}",
            phase="PHASE",
        )

    def summary(self, total_urls: int, peak_rss: int, output_path: Path):
        self.log(f"Dataset saved to: {output_path}", phase="DONE")
        self.log(f"Total URLs: {fmt_num(total_urls)}", phase="DONE")
        self.log(f"Peak memory: {fmt_mem(peak_rss)}", phase="DONE")


logger: Logger = Logger()


@contextmanager
def track_resources(phase_name: str):
    process = psutil.Process()
    start_time = time.perf_counter()
    start_mem = process.memory_info().rss
    peak_mem = start_mem
    start_cpu = process.cpu_times()

    logger.phase_start(phase_name)

    def _sample_peak():
        nonlocal peak_mem
        peak_mem = max(peak_mem, process.memory_info().rss)

    yield _sample_peak

    _sample_peak()
    end_time = time.perf_counter()
    end_cpu = process.cpu_times()

    duration = end_time - start_time
    memory_delta = peak_mem - start_mem
    cpu_user = end_cpu.user - start_cpu.user

    logger.phase_end(phase_name, duration, memory_delta, peak_mem, cpu_user)


async def fetch_sitemap_index(client: httpx.AsyncClient) -> list[str]:
    response = await client.get(SITEMAP_INDEX_URL)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    sitemap_locs = root.findall(".//sm:loc", ns)
    if not sitemap_locs:
        sitemap_locs = root.findall(".//loc")
    return [loc.text for loc in sitemap_locs if loc.text]


async def fetch_sitemap_to_records(client: httpx.AsyncClient, url: str) -> list[dict]:
    try:
        response = await client.get(url)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        url_elements = root.findall(".//sm:url", ns)

        results = []
        for url_elem in url_elements:
            loc = url_elem.find("sm:loc", ns)
            if loc is None or not loc.text:
                continue

            lastmod = url_elem.find("sm:lastmod", ns)
            changefreq = url_elem.find("sm:changefreq", ns)
            priority = url_elem.find("sm:priority", ns)

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


@dataclass
class ChunkAccumulator:
    records: list[dict] = field(default_factory=list)
    chunk_index: int = 0
    total_records: int = 0
    records_in_chunk: int = 0

    def add(self, new_records: list[dict], temp_dir: Path, fetched_at: str) -> None:
        for r in new_records:
            r["fetched_at"] = fetched_at
        self.records.extend(new_records)
        self.total_records += len(new_records)
        self.records_in_chunk += len(new_records)

        if self.records_in_chunk >= 25_000:
            self._flush(temp_dir)

    def flush_remaining(self, temp_dir: Path) -> None:
        if self.records:
            self._flush(temp_dir)

    def _flush(self, temp_dir: Path) -> None:
        df = pl.from_records(self.records)
        chunk_path = temp_dir / f"chunk_{self.chunk_index:04d}.parquet"
        df.write_parquet(chunk_path)
        logger.log(
            f"Wrote chunk {self.chunk_index}: {fmt_num(len(df))} rows "
            f"({fmt_num(self.total_records)} total)",
            phase="CHUNK",
        )
        self.chunk_index += 1
        self.records = []
        self.records_in_chunk = 0


async def fetch_and_stream_to_chunks(
    client: httpx.AsyncClient,
    sitemap_urls: list[str],
    temp_dir: Path,
    concurrency: int,
) -> int:
    fetched_at = datetime.now(timezone.utc).isoformat()
    acc = ChunkAccumulator()
    semaphore = anyio.Semaphore(concurrency)

    async def fetch_one(url: str) -> list[dict]:
        async with semaphore:
            return await fetch_sitemap_to_records(client, url)

    async def worker(url: str) -> None:
        records = await fetch_one(url)
        acc.add(records, temp_dir, fetched_at)

    async with anyio.create_task_group() as tg:
        for url in sitemap_urls:
            tg.start_soon(worker, url)

    acc.flush_remaining(temp_dir)
    return acc.chunk_index


@app.command()
def main(
    output: Path = Path("data/grokipedia_urls.parquet"),
    push: bool = False,
    hf_repo: str | None = None,
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
    """
    global logger
    logger = Logger(quiet=quiet)

    output.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="grokipedia_chunks_"))
    logger.log(f"Temp directory: {temp_dir}", phase="INFO")

    peak_rss = 0

    async def _fetch_phases() -> None:
        nonlocal peak_rss
        async with httpx.AsyncClient() as client:
            with track_resources("Phase 1: Fetch sitemap index"):
                sitemap_urls = await fetch_sitemap_index(client)
                logger.log(
                    f"Found {fmt_num(len(sitemap_urls))} child sitemaps", phase="INFO"
                )

            with track_resources("Phase 2: Fetch and chunk sitemaps"):
                num_chunks = await fetch_and_stream_to_chunks(
                    client, sitemap_urls, temp_dir, concurrency
                )
                logger.log(f"Created {fmt_num(num_chunks)} chunk files", phase="INFO")

        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)

    anyio.run(_fetch_phases)

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

    peak_rss = max(peak_rss, psutil.Process().memory_info().rss)

    df = pl.read_parquet(output)
    logger.summary(len(df), peak_rss, output)

    if not keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.log("Cleaned up temp directory", phase="DEBUG")

    if not push:
        return

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
            return

    hf_dataset = Dataset.from_parquet(str(output))
    hf_dataset.push_to_hub(repo_id, token=token, private=True)

    logger.log(f"Pushed to https://huggingface.co/datasets/{repo_id}", phase="DONE")


if __name__ == "__main__":
    app()
