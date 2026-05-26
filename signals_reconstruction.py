import os
import numpy as np
import pandas as pd

PARQUET_PATH = "C:/Users/bagon/OneDrive/Desktop/TactisRealization/signals.parquet"
OUT_ROOT = "data"
PROBLEM = "GearFaults8192"

EXPECTED_SIGNAL_LEN = 256000
WINDOW_SIZE = 8192    #norimas segmento ilgis
STRIDE = 8192         #persidengimas, jeigu norima nepersidengiancio tuomet irasyti window_size reiksme


def write_ts_univariate(df_windows, out_path, problem_name):
    class_values = sorted(df_windows["label"].astype(str).unique())

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"@problemName {problem_name}\n")
        f.write("@timestamps false\n")
        f.write("@univariate true\n")
        f.write("@classLabel true " + " ".join(class_values) + "\n")
        f.write("@data\n")

        for _, row in df_windows.iterrows():
            signal = np.asarray(row["signal"], dtype=np.float32)
            label = str(row["label"])
            signal_str = ",".join(map(str, signal.tolist()))
            f.write(f"{signal_str}:{label}\n")


def segment_signal(x, window_size, stride):
    x = np.asarray(x, dtype=np.float32)
    windows = []

    for start in range(0, len(x) - window_size + 1, stride):
        end = start + window_size
        windows.append(x[start:end])

    return windows


def build_window_dataset(df_signals, window_size, stride):
    rows = []

    for _, row in df_signals.iterrows():
        windows = segment_signal(row["signal"], window_size, stride)

        for w_idx, w in enumerate(windows):
            rows.append({
                "label": row["label"],
                "signal": w,
                "Feature": row["Feature"],
                "Bandymas": row["Bandymas"],
                "Apkrova(Nm)": row["Apkrova(Nm)"],
                "Sukiai(rpm)": row["Sukiai(rpm)"],
                "window_id": w_idx
            })

    return pd.DataFrame(rows)


def main():
    print("Reading parquet...")
    df = pd.read_parquet(PARQUET_PATH)

    required_cols = ["Feature", "Bandymas", "Apkrova(Nm)", "Sukiai(rpm)", "Value"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()

    print("Reconstructing signals...")
    grouped = (
        df.groupby(["Feature", "Bandymas", "Apkrova(Nm)", "Sukiai(rpm)"], sort=False)["Value"]
        .apply(lambda s: s.to_numpy(dtype=np.float32))
        .reset_index(name="signal")
    )

    grouped["length"] = grouped["signal"].apply(len)

    print("\nReconstructed signals:", len(grouped))
    print("Signal length counts:")
    print(grouped["length"].value_counts().sort_index())

    bad = grouped[grouped["length"] != EXPECTED_SIGNAL_LEN]
    if not bad.empty:
        print("\nSignals with unexpected length:")
        print(bad[["Feature", "Bandymas", "Apkrova(Nm)", "Sukiai(rpm)", "length"]])

    grouped = grouped[grouped["length"] == EXPECTED_SIGNAL_LEN].copy()
    grouped["label"] = grouped["Feature"].astype(str)

    train_signals = grouped[grouped["Bandymas"] == "1 bandymas"].copy()
    test_signals = grouped[grouped["Bandymas"] == "2 bandymas"].copy()

    print("\nComplete train signals:", len(train_signals))
    print("Complete test signals:", len(test_signals))

    print("\nSegmenting train...")
    train_windows = build_window_dataset(train_signals, WINDOW_SIZE, STRIDE)

    print("Segmenting test...")
    test_windows = build_window_dataset(test_signals, WINDOW_SIZE, STRIDE)

    print("\nTrain windows:", len(train_windows))
    print("Test windows:", len(test_windows))

    print("\nTrain class counts:")
    print(train_windows["label"].value_counts().sort_index())

    print("\nTest class counts:")
    print(test_windows["label"].value_counts().sort_index())

    out_dir = os.path.join(OUT_ROOT, PROBLEM)
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, f"{PROBLEM}_TRAIN.ts")
    test_path = os.path.join(out_dir, f"{PROBLEM}_TEST.ts")

    write_ts_univariate(train_windows[["signal", "label"]], train_path, PROBLEM)
    write_ts_univariate(test_windows[["signal", "label"]], test_path, PROBLEM)

    print("\nSaved files:")
    print(train_path)
    print(test_path)

if __name__ == "__main__":
    main()