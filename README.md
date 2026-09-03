# grokipedia-urls

A dataset of every URL from the Grokipedia sitemap, refreshed weekly via GitHub Actions and pushed to HuggingFace Hub.

## What it does

1. Fetches the sitemap index from `assets.grokipedia.com`
2. Concurrently scrapes all child sitemaps (20 parallel requests)
3. Writes a sorted parquet file with ~columns: `url`, `lastmod`, `changefreq`, `priority`, `sitemap_source`, `fetched_at`
4. Optionally pushes to [HuggingFace Hub](https://huggingface.co/datasets/caentzminger/grokipedia-urls)

## Setup

```bash
uv sync
cp .env.example .env  # add your HF_TOKEN
```

## Usage

```bash
# Collect locally only
uv run main.py

# Collect and push to HuggingFace
uv run main.py --push

# Force push (skip change detection)
uv run main.py --push --force-push
```

## Automation

A GitHub Actions workflow runs every Sunday at 00:00 UTC. To enable it, add `HF_TOKEN` as a repository secret.

## Development

```bash
just fmt      # format
just lint     # lint
just typecheck # type check
```
