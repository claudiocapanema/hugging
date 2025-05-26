

import geopandas as gpd
import numpy as np
import pandas as pd
import json
import ast

def get_hour(datetime):
    weekday = datetime.dt.weekday
    hour = datetime.dt.hour
    weekend = [24 if i >= 5 else 0 for i in weekday]

    return np.array([i + j for i, j in zip(hour, weekend)])

def get_category(categories):
    categories = [ast.literal_eval(i)[0]["name"] for i in categories]
    return np.array(categories)


import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers

    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_phi / 2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c  # in kilometers
    return round(distance, 2)



def calculate_distance_duration(data):

    data = data.sort_values("datetime")
    distances = []
    durations = []
    for i in range(1, len(data)):

        lat_a = data.iloc[i-1]["lat"]
        lon_a = data.iloc[i-1]["lng"]
        lat_b = data.iloc[i]["lat"]
        lon_b = data.iloc[i]["lng"]
        distance = haversine(lat_a, lon_a, lat_b, lon_b)
        dt_a = data.iloc[i-1]["datetime"]
        dt_b = data.iloc[i]["datetime"]
        duration = round((dt_b - dt_a).total_seconds() / 3600, 2)
        distances.append(distance)
        durations.append(duration)

    return pd.DataFrame({"hour": data.iloc[1::]["hour"], "datetime": data.iloc[1::]["datetime"], "distance": distances, "duration": durations, "sub_sub_category": data.iloc[1::]["sub_sub_category"]})


sample = 300000

us_states = gpd.read_file("us-state-boundaries/us-state-boundaries.shp")
us_states = us_states[us_states["name"] == "Texas"][["name", "geometry"]]
us_states["State"] = us_states["name"]
us_states = us_states[["State", "geometry"]]

spot_categories1 = gpd.read_file("gowalla_checkins/gowalla_spots_subset1.csv")[["id", "lng", "lat", "spot_categories"]]
spot_categories1 = gpd.GeoDataFrame(spot_categories1, geometry=gpd.points_from_xy(spot_categories1['lng'], spot_categories1['lat']))
spot_categories1["lat"] = spot_categories1["lat"].astype(float)
spot_categories1["lng"] = spot_categories1["lng"].astype(float)

# spot_categories2 = gpd.read_file("gowalla_checkins/gowalla_spots_subset2.csv", encoding='unicode_escape')
# gowalla_categories_structure = pd.read_json("gowalla_checkins/gowalla_category_structure.json")
category_sub_category_sub_sub_category = []
with open('gowalla_checkins/gowalla_category_structure.json', 'r') as file:
    gowalla_categories_structure = json.load(file)["spot_categories"]
# print(gowalla_categories_structure)
for category in gowalla_categories_structure:
    category_name = category["name"]

    for sub_category in category["spot_categories"]:
        sub_category_name = sub_category["name"]
        url = sub_category["url"]
        if len([sub_sub_category for sub_sub_category in sub_category["spot_categories"]]) > 0:
            for sub_sub_category in sub_category["spot_categories"]:
                sub_sub_category_name = sub_sub_category["name"]
                url = sub_sub_category["url"]
                category_sub_category_sub_sub_category.append([category_name, sub_category_name, sub_sub_category_name, url])
        else:
            category_sub_category_sub_sub_category.append(
                [category_name, sub_category_name, sub_category_name, url])

category_sub_category_sub_sub_category = pd.DataFrame(category_sub_category_sub_sub_category, columns=["category", "sub_category", "sub_sub_category", "url"])

print(us_states)
print(us_states.columns)
# print(gowalla_checkins)

# spot_categories = spot_categories1.join(spot_categories2.set_index("id"), on="id")

spot_categories1 = spot_categories1.sjoin(us_states)[["id", "lng", "lat", "spot_categories"]]
spot_categories1["placeid"] = spot_categories1["id"]
spot_categories1 = spot_categories1[["placeid", "lng", "lat", "spot_categories"]]
print(spot_categories1)
print(spot_categories1.columns)
gowalla_checkins = gpd.read_file("gowalla_checkins/gowalla_checkins.csv")
gowalla_checkins = gowalla_checkins.join(spot_categories1.set_index("placeid"), on="placeid", how="inner").sample(sample, random_state=42)
gowalla_checkins["datetime"] = pd.to_datetime(gowalla_checkins["datetime"], infer_datetime_format=True)
gowalla_checkins["hour"] = get_hour(gowalla_checkins["datetime"]).astype(int)
gowalla_checkins["sub_sub_category"] = get_category(gowalla_checkins["spot_categories"])
gowalla_checkins = gowalla_checkins[["userid", "datetime", "hour", "sub_sub_category", "lng", "lat", "spot_categories"]]
gowalla_checkins = gowalla_checkins.groupby("userid").apply(lambda e: calculate_distance_duration(e))
print(gowalla_checkins)
print(category_sub_category_sub_sub_category)
gowalla_checkins = gowalla_checkins.join(category_sub_category_sub_sub_category.set_index("sub_sub_category"), on="sub_sub_category", how="inner")
gowalla_checkins = gowalla_checkins.reset_index()[["userid", "datetime", "hour", "distance", "duration", "category", "sub_category", "sub_sub_category"]]
print(gowalla_checkins)
unique_categories = gowalla_checkins["category"].unique().tolist()
unique_sub_categories = gowalla_checkins["sub_category"].unique().tolist()
unique_sub_sub_category = gowalla_checkins["sub_sub_category"].unique().tolist()
unique_categories_dict = {unique_categories[i]: i for i in range(len(unique_categories))}
unique_sub_categories_dict = {unique_sub_categories[i]: i for i in range(len(unique_sub_categories))}
unique_sub_sub_categories_dict = {unique_sub_sub_category[i]: i for i in range(len(unique_sub_sub_category))}
gowalla_checkins["category_id"] = np.array([unique_categories_dict[category] for category in gowalla_checkins["category"].tolist()]).astype(int)
gowalla_checkins["sub_category_id"] = np.array([unique_sub_categories_dict[i] for i in gowalla_checkins["sub_category"].tolist()]).astype(int)
gowalla_checkins["sub_sub_category_id"] = np.array([unique_sub_sub_categories_dict[i] for i in gowalla_checkins["sub_sub_category"].tolist()]).astype(int)
gowalla_checkins.to_csv(f"gowalla_checkins_texas-sample-{sample}.csv", index=False)
print(f"quantidade de categories: {len(unique_categories)} \nQuantidade de sub categories: {len(unique_sub_categories)} \nQuantidade de sub sub categories: {len(unique_sub_sub_category)}")