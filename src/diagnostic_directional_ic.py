import numpy as np
import pandas as pd
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

from feature_engineering import build_features, build_labels, build_vol_regime_labels, load_mt5_csv, resample_to_4h

DATA_FILE = "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/XAUUSD_1H.csv"

def evaluate_directional_edge():
    print(f"Loading GOLD Data: {DATA_FILE}")
    df_raw = load_mt5_csv(DATA_FILE)
    df_4h = resample_to_4h(df_raw)
    
    print(f"Loaded {len(df_4h)} bars.")
    
    print("Building Features (Integrating Macro + Math Engine)...")
    features = build_features(df_4h)
    
    print("Building Directional Labels (Forward 4H Return)...")
    labels = build_labels(df_4h, horizon=1)
    
    print("Building Regime Labels (Vol > 0.8 proxy)...")
    regime_df = build_vol_regime_labels(df_4h)
    
    # Merge features, labels, and regime
    df_raw_merged = pd.concat([features, labels, regime_df["vol_regime"]], axis=1)
    df_merged = df_raw_merged.dropna()
    print(f"Merged dataset shape after dropping NaNs: {df_merged.shape}")
    
    # Filter for HIGH VOLATILITY regimes only
    df_high_vol = df_merged[df_merged["vol_regime"] == 1]
    print(f"Filtered for High Volatility Regime (Regime == 1): {df_high_vol.shape}")
    
    target = df_high_vol["label_ret"]
    
    # List of the new structural features we want to test
    test_features = [
        "coint_z_score",
        "hurst_90",
        "ou_theta",
        "ou_halflife",
        "ou_sigma",
        "cot_mm_net_long",
        "cot_mm_pct_oi",
        "macro_tips",
        "macro_tips_pct",
        "gld_volume_pct",
        "macro_dxy_pct"
    ]
    
    print("\n" + "="*60)
    print("CONDITIONAL IC ANALYSIS - 4H GOLD DIRECTION (HIGH VOL ONLY)")
    print("="*60)
    print("IC > 0.05 is considered an excellent institutional edge.")
    print("-" * 60)
    
    results = []
    for col in test_features:
        if col in df_high_vol.columns:
            feat_val = df_high_vol[col]
            # Spearman Rank Correlation (IC)
            ic, p_val = stats.spearmanr(feat_val, target)
            results.append({"Feature": col, "IC": ic, "P-Value": p_val})
        else:
            print(f"[Warning] {col} not found in features.")
            
    res_df = pd.DataFrame(results).sort_values(by="IC", key=abs, ascending=False)
    for _, row in res_df.iterrows():
        print(f"{row['Feature']:>20} | IC: {row['IC']:>7.4f} | P-Val: {row['P-Value']:.4f}")
        
    print("\n[Conditional Reversion Test]")
    # Let's test if the Fair Value Gap (Z-Score) works better during High Volatility Chop
    # Define Chop: Hurst < 0.45
    if "hurst_90" in df_high_vol.columns:
        chop_mask = df_high_vol["hurst_90"] < 0.45
        df_chop = df_high_vol[chop_mask]
        if len(df_chop) > 100 and "coint_z_score" in df_chop.columns:
            ic_chop, p_chop = stats.spearmanr(df_chop["coint_z_score"], df_chop["label_ret"])
            print(f"Cointegration IC (During High Vol Chop Regime): {ic_chop:.4f} (N={len(df_chop)})")
        else:
            print(f"Not enough samples for Conditional Chop test. N={len(df_chop)}")
            
if __name__ == "__main__":
    evaluate_directional_edge()
