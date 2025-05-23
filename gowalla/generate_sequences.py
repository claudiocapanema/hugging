import numpy as np
import pandas as pd
import json
import ast

np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'int': lambda x: str(x).replace(" ", ", ")})

def gen_sequences(gowalla_checkins, window, overlap):

    gowalla_checkins = gowalla_checkins.sort_values("datetime")
    X = []
    Y_category = []
    Y_sub_category = []
    Y_sub_sub_category = []
    for i in range(window, len(gowalla_checkins) - 1, int(window * overlap)):
        if i + window + 1 >= len(gowalla_checkins):
            continue
        X.append(gowalla_checkins[["category_id", "sub_category_id", "sub_sub_category_id", "hour", "distance", "duration"]].iloc[
                 i:i + window].to_numpy().astype(int).tolist())
        Y_category.append(gowalla_checkins.iloc[i + window + 1]["category_id"])
        Y_sub_category.append(gowalla_checkins.iloc[i + window + 1]["sub_category_id"])
        Y_sub_sub_category.append(gowalla_checkins.iloc[i + window + 1]["sub_sub_category_id"])

    return pd.DataFrame({"X": X, "Y_category": Y_category, "Y_sub_category": Y_sub_category, "Y_sub_sub_category": Y_sub_sub_category})

window = 4
overlap = 0.5

gowalla_checkins = pd.read_csv("gowalla_checkins_texas.csv").dropna()
gowalla_checkins["hour"] = gowalla_checkins["hour"].astype(int)
gowalla_checkins["category_id"] = gowalla_checkins["category_id"].astype(int)
gowalla_checkins["sub_category_id"] = gowalla_checkins["sub_category_id"].astype(int)
gowalla_checkins["sub_sub_category_id"] = gowalla_checkins["sub_sub_category_id"].astype(int)
gowalla_checkins = gowalla_checkins.groupby("userid").apply(lambda e: gen_sequences(e, window, overlap)).reset_index()[["X", "Y_category", "Y_sub_category", "Y_sub_sub_category"]]
gowalla_checkins["Y_category"] = gowalla_checkins["Y_category"].astype(int)
gowalla_checkins["Y_sub_category"] = gowalla_checkins["Y_sub_category"].astype(int)
gowalla_checkins["Y_sub_sub_category"] = gowalla_checkins["Y_sub_sub_category"].astype(int)
gowalla_checkins.to_csv(f"gowalla_checkins_texas_sequences_window_{window}_overlap_{overlap}.csv", index=False)
print(gowalla_checkins)

