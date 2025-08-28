import requests
import os
import pandas as pd
import time
import re
from bs4 import BeautifulSoup

def slugify(title):
    # Convert episode title to URL slug format.
    title = title.lower()
    title = re.sub(r"[^\w\s-]", "", title)  # Remove punctuation
    title = re.sub(r"[\s_]+", "-", title)   # Replace spaces/underscores with dashes
    title = re.sub(r"-{2,}", "-", title)    # Collapse multiple dashes into one (Ep 487/488/542/562/563 fix)
    return title.strip("-")


def fetch_summary_from_url(url, episode_id):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title_tag = soup.select_one(".episode-title") or soup.select_one("h1.node-title")
        meta_tag = soup.find("meta", attrs={"name": "description"})

        if not title_tag or not meta_tag or not meta_tag.get("content"):
            print(f"Missing content for episode {episode_id}")
            return None

        return {
            "episode_id": episode_id,
            "url": url,
            "title": title_tag.text.strip(),
            "summary": meta_tag["content"].strip()
        }

    except Exception as e:
        print(f" Error scraping {url}: {e}")
        return None


if __name__ == "__main__":
    episode_df = pd.read_csv("data/episode_info_clean.csv")
    summaries = []

    for _, row in episode_df.iterrows():
        episode_id = int(row["episode_number"])
        raw_title = str(row["title"])
        slug = slugify(raw_title)
        url = f"https://www.thisamericanlife.org/{episode_id}/{slug}"
        print(f"Scraping {url}...")
        result = fetch_summary_from_url(url, episode_id)
        if result:
            summaries.append(result)
        time.sleep(0.5)

    # Ensure /data directory exists
    os.makedirs("data", exist_ok=True)

    # Save to CSV
    df = pd.DataFrame(summaries)
    output_path = "data/episode_summaries_full.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved summaries to {output_path}")
