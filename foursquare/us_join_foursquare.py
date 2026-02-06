import pandas as pd
from collections import Counter

BASE = "/media/gustavo/e5069d0b-8c3d-4850-8e10-319862c4c082/home/claudio/Downloads/dataset_TIST2015/"
CHECKINS_PATH = f"{BASE}dataset_TIST2015_Checkins.txt"
POIS_PATH = f"{BASE}dataset_TIST2015_POIs.txt"
CITIES_PATH = f"{BASE}dataset_TIST2015_Cities.txt"

CHUNK_SIZE = 1_000_000

# ===============================
# Load POIs
# ===============================
pois = pd.read_csv(
    POIS_PATH,
    sep="\t",
    header=None,
    names=["venue_id", "latitude", "longitude", "category", "country_code"]
)

# ===============================
# Load Cities → country map
# ===============================
cities = pd.read_csv(
    CITIES_PATH,
    sep="\t",
    header=None,
    names=["city", "city_lat", "city_lon", "country_code", "country", "timezone"]
)

country_map = (
    cities[["country_code", "country"]]
    .drop_duplicates()
    .set_index("country_code")["country"]
    .to_dict()
)

# ===============================
# Count check-ins per country
# ===============================
country_counter = Counter()

for chunk in pd.read_csv(
    CHECKINS_PATH,
    sep="\t",
    header=None,
    names=["user_id", "venue_id", "utc_time", "timezone_offset"],
    chunksize=CHUNK_SIZE
):
    df = chunk.merge(
        pois[["venue_id", "country_code"]],
        on="venue_id",
        how="inner"
    )

    df["country"] = df["country_code"].map(country_map)

    country_counter.update(df["country"].dropna().values)

print("🌍 Top 10 países por check-ins:")
for country, cnt in country_counter.most_common(10):
    print(country, cnt)

top_country = country_counter.most_common(1)[0][0]
print(f"\n🏆 País com mais check-ins: {top_country}")
top_country = "United States"

import pandas as pd

BASE = "/media/gustavo/e5069d0b-8c3d-4850-8e10-319862c4c082/home/claudio/Downloads/dataset_TIST2015/"
CHECKINS_PATH = f"{BASE}dataset_TIST2015_Checkins.txt"
POIS_PATH = f"{BASE}dataset_TIST2015_POIs.txt"
CITIES_PATH = f"{BASE}dataset_TIST2015_Cities.txt"
OUTPUT_PATH = f"{BASE}dataset_tist2015_{top_country.replace(' ', '_')}.csv"

CHUNK_SIZE = 1_000_000

# ===============================
# Load POIs
# ===============================
pois = pd.read_csv(
    POIS_PATH,
    sep="\t",
    header=None,
    names=["venue_id", "latitude", "longitude", "category", "country_code"]
)

# ===============================
# Load Cities → country map
# ===============================
cities = pd.read_csv(
    CITIES_PATH,
    sep="\t",
    header=None,
    names=["city", "city_lat", "city_lon", "country_code", "country", "timezone"]
)

country_map = (
    cities[["country_code", "country"]]
    .drop_duplicates()
    .set_index("country_code")["country"]
    .to_dict()
)

# ===============================
# Stream + filter
# ===============================
first_chunk = True

for chunk in pd.read_csv(
    CHECKINS_PATH,
    sep="\t",
    header=None,
    names=["user_id", "venue_id", "utc_time", "timezone_offset"],
    chunksize=CHUNK_SIZE
):
    # Parse UTC time
    chunk["utc_time"] = pd.to_datetime(
        chunk["utc_time"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce"
    )
    chunk = chunk.dropna(subset=["utc_time"])

    # Local time
    chunk["local_time"] = (
        chunk["utc_time"]
        + pd.to_timedelta(chunk["timezone_offset"], unit="m")
    )

    # Temporal features
    chunk["local_hour"] = chunk["local_time"].dt.hour
    chunk["local_dayofweek"] = chunk["local_time"].dt.dayofweek
    chunk["local_is_weekend"] = chunk["local_dayofweek"].isin([5, 6]).astype(int)

    # Join com POIs
    df = chunk.merge(
        pois,
        on="venue_id",
        how="inner"
    )

    # Country
    df["country"] = df["country_code"].map(country_map)

    # 🔥 FILTRO CRÍTICO
    df = df[df["country"] == top_country]

    if df.empty:
        continue

    # Colunas finais
    df = df[
        [
            "user_id",
            "venue_id",
            "category",
            "utc_time",
            "local_time",
            "local_hour",
            "local_dayofweek",
            "local_is_weekend",
            "latitude",
            "longitude",
            "timezone_offset",
            "country"
        ]
    ]

    # Append
    df.to_csv(
        OUTPUT_PATH,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False
    print(f"✅ Chunk salvo ({top_country}): {len(df)} registros")
