import pandas as pd

# ===============================
# Paths
# ===============================
BASE = "/media/gustavo/e5069d0b-8c3d-4850-8e10-319862c4c082/home/claudio/Downloads/dataset_TIST2015/"
OUTPUT_PATH = f"{BASE}dataset_tist2015_United_States.csv"

df = pd.read_csv(OUTPUT_PATH)

print(df)