

import geopandas as gpd
import pandas as pd
import json



us_states = gpd.read_file("us-state-boundaries/us-state-boundaries.shp")
us_states = us_states[us_states["name"] == "Texas"][["name", "geometry"]]
us_states["State"] = us_states["name"]
us_states = us_states[["State", "geometry"]]
# gowalla_checkins = gpd.read_file("gowalla_checkins/gowalla_checkins.csv")
spot_categories1 = gpd.read_file("gowalla_checkins/gowalla_spots_subset1.csv")[["id", "lng", "lat", "spot_categories"]]
spot_categories1 = gpd.GeoDataFrame(spot_categories1, geometry=gpd.points_from_xy(spot_categories1['lng'], spot_categories1['lat']))
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
print(spot_categories1)
spot_categories1 = spot_categories1.sjoin(us_states)
print(spot_categories1)