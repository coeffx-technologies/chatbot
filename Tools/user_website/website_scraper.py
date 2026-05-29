"""
Website Intelligence Scraper
------------------------------
Give it any website URL → it searches Google for key pages
(pricing, features, about, etc.) → scrapes the top results
→ saves everything into a clean .txt file.

Uses:
  - apify/google-search-scraper  : to find the right pages via Google
  - apify/website-content-crawler: to extract clean text from those pages


"""

import os
import time
from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")

# What we want to learn about any website
SEARCH_INTENTS = [
    "pricing",
    "features",
    "about",
    "customers OR case studies",
    "solutions",
]

# How many Google results to scrape per query (we take top 1)
RESULTS_PER_QUERY = 1


# ── Step 1: Google Search → find the right URLs ───────────────────────────────

def find_pages_via_google(client: ApifyClient, website_url: str) -> list[str]:
    """
    Runs site:example.com <intent> queries on Google.
    Returns a deduplicated list of URLs found.
    """

    # Build all queries like: site:kissflow.com pricing
    domain = website_url.replace("https://", "").replace("http://", "").rstrip("/")
    queries = [f"site:{domain} {intent}" for intent in SEARCH_INTENTS]

    print(f"\n[1/3] Running {len(queries)} Google searches for: {domain}")
    print("      Queries:", queries[:3], "...")

    run_input = {
        "queries": "\n".join(queries),   # one query per line
        "maxPagesPerQuery": 1,           # just the first results page
        "resultsPerPage": RESULTS_PER_QUERY,
        "outputFormats": ["captions"],   # we only need URLs, not full HTML
    }

    run = client.actor("apify/google-search-scraper").call(run_input=run_input)
    results = client.dataset(run["defaultDatasetId"]).list_items().items

    # Pull out URLs from organic results, deduplicate, keep only same domain
    found_urls = []
    seen = set()

    # Always include the homepage directly
    found_urls.insert(0, website_url)
    seen.add(website_url)

    for result in results:
        organic = result.get("organicResults", [])
        for item in organic[:RESULTS_PER_QUERY]:
            url = item.get("url", "")
            if url and url not in seen and domain in url:
                found_urls.append(url)
                seen.add(url)

    print(f"      Found {len(found_urls)} unique URLs to scrape")
    for url in found_urls:
        print(f"        → {url}")

    return found_urls


# ── Step 2: Scrape each URL → extract clean text ──────────────────────────────

def scrape_pages(client: ApifyClient, urls: list[str]) -> list[dict]:
    """
    Feeds URLs into website-content-crawler.
    Returns list of {url, title, text} dicts.
    """

    print(f"\n[2/3] Scraping {len(urls)} pages with website-content-crawler...")

    run_input = {
        "startUrls": [{"url": url} for url in urls],
        "maxCrawlPages": len(urls),   # only scrape what we found, no extra crawling
        "maxCrawlDepth": 0,           # depth 0 = don't follow links, just these pages
        "crawlerType": "cheerio",     # fast HTML scraper (use "playwright:firefox" for JS-heavy sites)
        "outputFormats": ["markdown"],# clean readable output
    }

    run = client.actor("apify/website-content-crawler").call(run_input=run_input)
    items = client.dataset(run["defaultDatasetId"]).list_items().items

    pages = []
    for item in items:
        pages.append({
            "url":   item.get("url", ""),
            "title": item.get("metadata", {}).get("title") or item.get("title", "No title"),
            "text":  item.get("markdown") or item.get("text", ""),
        })

    print(f"      Successfully scraped {len(pages)} pages")
    return pages


# ── Step 3: Write everything to a .txt file ───────────────────────────────────

def save_to_txt(pages: list[dict], website_url: str, output_path: str = None) -> str:
    """
    Combines all scraped pages into one readable .txt file.
    Returns the file path.
    """

    domain = website_url.replace("https://", "").replace("http://", "").rstrip("/")
    domain_clean = domain.replace(".", "_").replace("/", "_")

    if output_path is None:
        output_path = f"{domain_clean}_scraped.txt"

    print(f"\n[3/3] Saving results to: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:

        # Header
        f.write("=" * 70 + "\n")
        f.write(f"WEBSITE INTELLIGENCE REPORT\n")
        f.write(f"Website : {website_url}\n")
        f.write(f"Scraped : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pages   : {len(pages)}\n")
        f.write("=" * 70 + "\n\n")

        # Table of contents
        f.write("PAGES SCRAPED:\n")
        for i, page in enumerate(pages, 1):
            f.write(f"  {i}. {page['title']}\n")
            f.write(f"     {page['url']}\n")
        f.write("\n" + "-" * 70 + "\n\n")

        # Each page's content
        for i, page in enumerate(pages, 1):
            f.write(f"PAGE {i}: {page['title']}\n")
            f.write(f"URL: {page['url']}\n")
            f.write("-" * 50 + "\n\n")
            f.write(page["text"].strip())
            f.write("\n\n" + "=" * 70 + "\n\n")

    print(f"      Done! File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    return output_path


# ── Main function: give URL → get txt file ────────────────────────────────────

def scrape_website(website_url: str, output_path: str = None) -> str:
    """
    Main entry point.

    Args:
        website_url : The website to scrape e.g. "https://kissflow.com"
        output_path : Where to save the .txt file (optional, auto-named if not given)

    Returns:
        Path to the output .txt file
    """

    if not APIFY_TOKEN:
        raise ValueError("APIFY_API_TOKEN not found in environment / .env file")

    print(f"\n{'='*50}")
    print(f"  Scraping: {website_url}")
    print(f"{'='*50}")

    client = ApifyClient(APIFY_TOKEN)

    # Step 1 — Google search to find the right pages
    urls = find_pages_via_google(client, website_url)

    # Step 2 — Scrape those pages
    pages = scrape_pages(client, urls)

    if not pages:
        raise RuntimeError("No pages were scraped. Check your Apify token and actor permissions.")

    # Step 3 — Save to txt
    output_file = save_to_txt(pages, website_url, output_path)

    print(f"\n✅ Complete! Output saved to: {output_file}\n")
    return output_file


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Change this URL to scrape any website
    result = scrape_website("https://kissflow.com/")
    print(f"File: {result}")