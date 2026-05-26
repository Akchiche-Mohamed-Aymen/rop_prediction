import numpy as np
def create_features(df):
    df = df.copy()

    eps = 1e-6  # to avoid division by zero

    # =========================
    # 1. Core interactions
    # =========================
    df["wob_rpm_ratio"] = df["wob"] / (df["rpm"] + eps)
    df["wob_rpm_prod"] = df["wob"] * df["rpm"]

    df["flow_rpm_ratio"] = df["flow_in"] / (df["rpm"] + eps)
    df["flow_wob_prod"] = df["flow_in"] * df["wob"]

    # =========================
    # 2. Depth interactions
    # =========================
    df["depth_tvd_wob"] = df["depth_tvd"] * df["wob"]
    df["depth_tvd_rpm"] = df["depth_tvd"] * df["rpm"]

    df["depth_tmd_wob"] = df["depth_tmd"] * df["wob"]
    df["depth_tmd_rpm"] = df["depth_tmd"] * df["rpm"]
  
    # =========================
    # 4. Combined energy feature
    # =========================
    df["energy_proxy"] = df["wob"] * df["rpm"] * df["flow_in"]

    df["wob_flow_ratio"] = df["wob"] / (df["flow_in"] + eps)
    df["rpm_flow_ratio"] = df["rpm"] / (df["flow_in"] + eps)

    # =========================
    # 6. Log transforms
    # =========================
    df["log_wob"] = np.log1p(df["wob"])
    df["log_rpm"] = np.log1p(df["rpm"])
    df["log_flow_in"] = np.log1p(df["flow_in"])

    return df