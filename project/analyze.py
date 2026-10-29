import pandas as pd

def analyze():
    df = pd.read_parquet("output/dataset.parquet")
    df['is_outlier'] = df['speed_kmh'] > 12
    dv = pd.DataFrame({
        "question": [
            "Сколько точек и сколько треков в итоговом датасете",
            "Общая длина всех треков (км) и длина каждого региона",
            "Какой трек самый длинный и какой самый быстрый (по средней скорости, км/ч)",
            "Средняя частота шагов по сезонам",
            "Суммарный набор высоты по каждому треку; трек с максимальным набором",
            "Час суток, в который туристы идут быстрее всего (по средней скорости)",
            "Коэффициент корреляции между температурой и скоростью (corr) по всем точкам",
            "Количество «подозрительных» точек, у которых speed_kmh больше 12",
            ],
        "answer": [
            f"Точек: {df.shape[0]}, треков: {len(df['track_id'].drop_duplicates())}",
            f"Общая длина: {round(df['dist_m'].sum() / 1000, 2)} км, длина каждого региона: {df.groupby('region')['dist_m'].sum().div(1000).round(2).to_dict()}",
            f"Самый длинный трек: {df.groupby('name')['dist_m'].sum().idxmax()}, самый быстрый трек: {df.groupby('name')['speed_kmh'].mean().idxmax()}",
            f"{df.groupby('season')['cadence_spm'].mean().round(2).to_dict()}",
            f"Набор высоты по каждому треку: {df.groupby('name')['elevation_m'].apply(lambda x: x.diff().clip(lower=0).sum()).round(2).to_dict()}, трек с максимальным набором: {df.groupby('name')['elevation_m'].apply(lambda x: x.diff().clip(lower=0).sum()).idxmax()}",
            f"{df.groupby('hour')['speed_kmh'].mean().idxmax()} часов",
            f"{round(df['temperature_c'].corr(df['speed_kmh']), 2)}",
            f"Всего точек: {df['is_outlier'].sum()}, по регионам: {df.groupby('region')['is_outlier'].mean().round(2).to_dict()}",
            ]
        })
    
    summary = df.groupby(["track_id", "name", "region", "season"]).agg(
        points      = ("timestamp", "size"),
        distance_km = ("dist_m", lambda s: s.sum() / 1000),
        duration_h  = ("dt_s", lambda s: s.sum() / 3600),
        diff_m      = ("elevation_m", lambda s: s.diff().clip(lower=0).sum()),
        avg_cadence = ("cadence_spm", "mean"),
        avg_temp    = ("temperature_c", "mean"),
    ).round(1).reset_index()
    
    df.to_parquet("output/dataset.parquet", index=False)
    dv.to_csv("output/answers.csv", index=False)
    summary.to_csv("output/summary.csv", index=False)
    print(dv)

if __name__ == "__main__":
    analyze()
