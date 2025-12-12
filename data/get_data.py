import requests
import pandas as pd
import time
import os

# ==============================
# Configuration
# ==============================
OUTPUT_CSV = "arabic_personalities_full.csv"
LIMIT = 1000        # Results per query
MAX_RESULTS = 50000
SAVE_INTERVAL = 5000
SLEEP_BETWEEN_BATCHES = 3

# ==============================
# Load previous progress
# ==============================
if os.path.exists(OUTPUT_CSV):
    print(f"📂 Found existing file: {OUTPUT_CSV}, resuming progress...")
    df_existing = pd.read_csv(OUTPUT_CSV)
    all_people = df_existing.to_dict(orient="records")
    offset = len(all_people)
    print(f"Resuming from offset {offset} records.")
else:
    all_people = []
    offset = 0

# ==============================
# Helper: Fetch Wikipedia Description
# ==============================
def get_wikipedia_description(name):
    """Fetches English description from Wikipedia API."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{name.replace(' ', '_')}"
        headers = {"User-Agent": "AkinatorDataCollector/1.0 (Yousef)"}
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 200:
            data = response.json()
            return data.get("extract", "")
        else:
            return ""
    except Exception:
        return ""

# ==============================
# Helper: Fetch batch from Wikidata
# ==============================
def fetch_batch(offset):
    query = f"""
    SELECT ?person ?personLabel ?genderLabel ?countryLabel ?occupationLabel
           ?birthDate ?deathDate ?image
    WHERE {{
      ?person wdt:P31 wd:Q5.
      ?person wdt:P27 ?country.
      VALUES ?country {{
        wd:Q79 wd:Q851 wd:Q916 wd:Q958 wd:Q843 wd:Q878 wd:Q810 wd:Q817
        wd:Q921 wd:Q1016 wd:Q1011 wd:Q846 wd:Q805 wd:Q813 wd:Q796 wd:Q800
        wd:Q805 wd:Q1028 wd:Q928 wd:Q1037 wd:Q117 wd:Q810 wd:Q817 wd:Q912 wd:Q794
      }}
      OPTIONAL {{ ?person wdt:P21 ?gender. }}
      OPTIONAL {{ ?person wdt:P106 ?occupation. }}
      OPTIONAL {{ ?person wdt:P569 ?birthDate. }}
      OPTIONAL {{ ?person wdt:P570 ?deathDate. }}
      OPTIONAL {{ ?person wdt:P18 ?image. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {LIMIT}
    OFFSET {offset}
    """

    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": "AkinatorDataCollector/1.0 (Yousef)"}

    for attempt in range(10):
        try:
            response = requests.get(url, params={"format": "json", "query": query}, headers=headers, timeout=60)

            if response.status_code == 429:
                print("⚠️ Rate limit reached (HTTP 429). Waiting 60s...")
                time.sleep(60)
                continue

            if response.status_code != 200:
                print(f"⚠️ HTTP {response.status_code}. Retrying in 15s...")
                time.sleep(15)
                continue

            data = response.json()
            people = []

            for item in data["results"]["bindings"]:
                person = {
                    "name": item.get("personLabel", {}).get("value", "").strip(),
                    "gender": item.get("genderLabel", {}).get("value", "").strip(),
                    "country": item.get("countryLabel", {}).get("value", "").strip(),
                    "occupation": item.get("occupationLabel", {}).get("value", "").strip(),
                    "birth_date": item.get("birthDate", {}).get("value", "").strip(),
                    "death_date": item.get("deathDate", {}).get("value", "").strip(),
                    "image_url": item.get("image", {}).get("value", "").strip(),
                }

                # Check required fields
                required = ["name", "gender", "country", "occupation", "birth_date"]
                if any(not person[f] for f in required):
                    continue

                # Fetch Wikipedia description
                description = get_wikipedia_description(person["name"])
                if not description:
                    continue  # Skip if no description available

                person["description"] = description
                people.append(person)

            return people

        except Exception as e:
            print(f"⚠️ Error fetching batch: {e}. Retrying in 20s...")
            time.sleep(20)

    print("❌ Failed after several retries, skipping this batch.")
    return []
# ==============================
# Main Loop (safer: save every batch + save on interrupt)
# ==============================
try:
    while len(all_people) < MAX_RESULTS:
        print(f"\n📡 Fetching batch starting at offset {offset}...")
        batch = fetch_batch(offset)

        if not batch:
            print("🚫 No more results. Possibly end of data.")
            break

        all_people.extend(batch)
        offset += LIMIT
        print(f"✅ Collected total: {len(all_people)}")

        # --- Save progress after every batch (safer) ---
        pd.DataFrame(all_people).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"💾 Saved progress: {len(all_people)} records to {OUTPUT_CSV}")

        # Respect rate limit
        time.sleep(SLEEP_BETWEEN_BATCHES)

except KeyboardInterrupt:
    print("\n⛔ Interrupted by user (KeyboardInterrupt). Saving current progress...")
    pd.DataFrame(all_people).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✔️ Saved {len(all_people)} records to {OUTPUT_CSV}. Exiting.")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}. Saving current progress before exit...")
    pd.DataFrame(all_people).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✔️ Saved {len(all_people)} records to {OUTPUT_CSV}.")
    raise
else:
    # Final save if loop ended normally
    pd.DataFrame(all_people).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n🎯 Done! Total {len(all_people)} records saved to {OUTPUT_CSV}.")
