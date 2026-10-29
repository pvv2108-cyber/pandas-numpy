import numpy as np
import pandas as pd
 
def prepare():
    tracks  = pd.read_csv("data/ml2_tracks.csv")
    points  = pd.read_csv("data/ml2_points.csv", sep=";")
    weather = pd.read_csv("data/ml2_weather.csv")
    
    points = points.drop_duplicates()
    tracks = tracks.drop_duplicates()
    weather = weather.drop_duplicates()
    
    tracks["region"] = tracks["region"].str.strip().str.capitalize()
    points["elevation_m"] = pd.to_numeric(points["elevation_m"], errors="coerce")
    
    dubl = points["timestamp"]
    format1 = pd.to_datetime(dubl, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    format2 = pd.to_datetime(dubl, format="%d.%m.%Y %H:%M", errors="coerce")
    points["timestamp"] = format2.fillna(format1)
    
    points.loc[points["cadence_spm"] <= 0, "cadence_spm"] = np.nan
    points.loc[points["elevation_m"] > 6000, "elevation_m"] = np.nan
    
    points["elevation_m"] = (
        points.groupby("track_id")["elevation_m"]
              .transform(lambda s: s.interpolate(limit_direction="both"))
    )
    
    median_by_track = points.groupby("track_id")["cadence_spm"].transform("median")
    points["cadence_spm"] = points["cadence_spm"].fillna(median_by_track)
    points = points.sort_values(["track_id", "timestamp"]).reset_index(drop=True)
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6_371_000
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        a = (np.sin((lat2 - lat1) / 2) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
        return 2 * R * np.arcsin(np.sqrt(a))
    
    by_track = points.groupby("track_id")
    points["dist_m"] = haversine(by_track["lat"].shift(), by_track["lon"].shift(), points["lat"], points["lon"]).fillna(0)
    points["dt_s"] = by_track["timestamp"].diff().dt.total_seconds().fillna(0)
    points["speed_kmh"] = np.where(points["dt_s"] > 0, points["dist_m"] / points["dt_s"] * 3.6, 0)
    
    seasons = {12: "зима", 1: "зима", 2: "зима",
               3: "весна", 4: "весна", 5: "весна",
               6: "лето", 7: "лето", 8: "лето",
               9: "осень", 10: "осень", 11: "осень"}
    
    points["hour"]   = points["timestamp"].dt.hour
    points["month"]  = points["timestamp"].dt.month
    points["season"] = points["month"].map(seasons) 
    
    df = points.merge(tracks[["track_id", "name", "region"]], on="track_id", how="left", validate="many_to_one")
    weather["time"] = pd.to_datetime(weather["time"])
    
    df = df.sort_values("timestamp")
    weather = weather.sort_values("time")
    
    df = pd.merge_asof(df, weather,
                       left_on="timestamp", right_on="time",
                       by="region",
                       direction="nearest",
                       tolerance=pd.Timedelta("2h"))
    df = df.drop(columns="time")
    print(df)
    df.to_parquet("output/dataset.parquet", index=False)
    return df

if __name__ == "__main__":
    prepare()
