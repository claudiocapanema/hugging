

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
    # Earth radius in kilometers (use 3958.8 for miles)
    R = 6371.0

    # Convert coordinates from degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c  # *

    return distance


def calculate_distance_duration(data):

    for i in range(1, len(data)):

        lat_a = data[i-1]["lat"]
        lon_a = data[i-1]["lon"]
        lat_b = data[i]["lat"]
        lon_b = data[i]["lon"]
        distance = haversine(lat_a, lon_b, lat_a, lon_b)




us_states = gpd.read_file("us-state-boundaries/us-state-boundaries.shp")
us_states = us_states[us_states["name"] == "Texas"][["name", "geometry"]]
us_states["State"] = us_states["name"]
us_states = us_states[["State", "geometry"]]

spot_categories1 = gpd.read_file("gowalla_checkins/gowalla_spots_subset1.csv")[["id", "lng", "lat", "spot_categories"]]
spot_categories1 = gpd.GeoDataFrame(spot_categories1, geometry=gpd.points_from_xy(spot_categories1['lng'], spot_categories1['lat']))
gowalla_checkins = gpd.read_file("gowalla_checkins/gowalla_checkins.csv")
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
        for sub_sub_category in sub_category["spot_categories"]:
            sub_sub_category_name = sub_sub_category["name"]
            url = sub_sub_category["url"]
            category_sub_category_sub_sub_category.append([category_name, sub_category_name, sub_sub_category_name, url])

category_sub_category_sub_sub_category = pd.DataFrame(category_sub_category_sub_sub_category, columns=["Category", "Sub_category", "Sub_sub_category", "url"])
print(category_sub_category_sub_sub_category)

print(us_states)
print(us_states.columns)
# print(gowalla_checkins)

# spot_categories = spot_categories1.join(spot_categories2.set_index("id"), on="id")

spot_categories1 = spot_categories1.sjoin(us_states)[["id", "lng", "lat", "spot_categories"]]
spot_categories1["placeid"] = spot_categories1["id"]
spot_categories1 = spot_categories1[["placeid", "lng", "lat", "spot_categories"]]
print(spot_categories1)
print(spot_categories1.columns)
gowalla_checkins = gowalla_checkins.join(spot_categories1.set_index("placeid"), on="placeid", how="inner")
gowalla_checkins["datetime"] = pd.to_datetime(gowalla_checkins["datetime"], infer_datetime_format=True)
gowalla_checkins["hour"] = get_hour(gowalla_checkins["datetime"])
gowalla_checkins["category"] = get_category(gowalla_checkins["spot_categories"])
gowalla_checkins = gowalla_checkins[["user_id", "datetime", "hour", "category", "lng", "lat", "spot_categories"]]
print(gowalla_checkins)