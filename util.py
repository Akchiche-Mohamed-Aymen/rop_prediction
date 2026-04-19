def create_features(df):
    df["hydraulic_power"] = df["pump_pressure"] * df["flow_in"]
    df["mechanical_power"] = df["torque"] * df["rpm"]
    df["flow_delta"] = df["flow_in"] - df["flow_out"]
    df["temp_diff"] = abs(df["temp_in"] - df["temp_out"])
    return df