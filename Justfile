fmt:
    uv run --dev ruff format main.py

typecheck:
    uv run --dev ty check main.py

lint:
    uv run --dev ruff check main.py

test:
    uv run main.py --help

smoke:
    uv run python -c "import httpx, xml.etree.ElementTree as ET; r=httpx.get('https://assets.grokipedia.com/sitemap/sitemap-index.xml', timeout=30); r.raise_for_status(); root=ET.fromstring(r.text); ns={'sm':'http://www.sitemaps.org/schemas/sitemap/0.9'}; locs=root.findall('.//sm:loc', ns) or root.findall('.//loc'); assert len(locs) > 200, len(locs); c=httpx.get(locs[0].text, timeout=30); c.raise_for_status(); assert '<loc>https://grokipedia.com/page/' in c.text; print(f'smoke ok: {len(locs)} sitemaps, first child {len(c.text)} bytes')"

collect:
    uv run main.py

collect-push:
    uv run main.py --push
