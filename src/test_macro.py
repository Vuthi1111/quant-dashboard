import pandas as pd
from feature_engineering import build_features, load_mt5_csv

df = load_mt5_csv("/Users/macos/Documents/ALGO/03_Data/raw/NAS100/1h_data.csv")
print("Original DF shape:", df.shape)
# Just test on the last 1000 rows to speed it up
df = df.iloc[-1000:]

features = build_features(df)
print("Features DF shape:", features.shape)

macro_cols = ["macro_vix", "macro_vix_pct", "macro_dxy", "macro_dxy_pct", 
              "macro_tnx", "macro_tnx_pct", "macro_hyg", "macro_hyg_pct"]

for col in macro_cols:
    if col in features.columns:
        print(f"[{col}] valid count: {features[col].notna().sum()}, max: {features[col].max():.2f}")
    else:
        print(f"[{col}] MISSING!")

print("TEST COMPLETE")
