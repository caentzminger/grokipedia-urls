# grokipedia-urls

[![Run Grokipedia URL Collector](https://github.com/caentzminger/grokipedia-urls/actions/workflows/update-uls-dataset.yml/badge.svg)](https://github.com/caentzminger/grokipedia-urls/actions/workflows/update-uls-dataset.yml)

A dataset of every URL from the [Grokipedia](https://grokipedia.com) sitemap, refreshed weekly via GitHub Actions and pushed to HuggingFace Hub.

## Dataset

**[caentzminger/grokipedia-urls](https://huggingface.co/datasets/caentzminger/grokipedia-urls)** on HuggingFace Hub.

| Column | Type | Description |
|--------|------|-------------|
| `url` | string | The page URL |
| `lastmod` | date | Last modified date |
| `changefreq` | enum | Change frequency (always, hourly, daily, weekly, monthly, yearly, never) |
| `priority` | float32 | Sitemap priority |
| `sitemap_source` | string | Which child sitemap this URL came from |
| `fetched_at` | datetime | When the URL was collected |

## Usage

### Load from HuggingFace

```python
from datasets import load_dataset

ds = load_dataset("caentzminger/grokipedia-urls", split="train")
ds.to_pandas()  # or .to_arrow(), etc.
```

### Load from parquet

```python
import polars as pl

df = pl.read_parquet("data/grokipedia_urls.parquet")
```

## Local development

```bash
git clone https://github.com/caentzminger/grokipedia-urls.git
cd grokipedia-urls
uv sync
```

### CLI

```bash
# Collect locally only
uv run main.py

# Collect and push to HuggingFace
uv run main.py --push

# Force push (skip change detection)
uv run main.py --push --force-push

# See all options
uv run main.py --help
```

### Development commands

```bash
just fmt       # format
just lint      # lint
just typecheck # type check
```

## License

[MIT](LICENSE)
