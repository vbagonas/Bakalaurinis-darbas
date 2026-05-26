import os
import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib.mlab as mlab
PARQUET_PATH = "C:/Users/bagon/OneDrive/Desktop/TactisRealization/signals.parquet"
OUT_ROOT = "data"
PROBLEM = "GearFaults_PSD_1024_v2"
 
EXPECTED_SIGNAL_LEN = 256000
SEGMENT_SIZE = 1024 #segmento ilgis
STRIDE = 1024  #segmentai nepersidengia

NPERSEG = 512    #welch lango ilgis
NOVERLAP = 256         #welch persidengimas

SAMPLE_RATE = 51200  #signalo daznis
 
 
def compute_psd(segment: np.ndarray, fs: int, nperseg: int, noverlap: int) -> np.ndarray:
    psd, _ = mlab.psd(
        segment,
        Fs=fs,
        NFFT=nperseg,
        noverlap=noverlap,  
        detrend=mlab.detrend_mean 
    )
    return (10 * np.log10(psd + 1e-10)).astype(np.float32)
 
 
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
 
 
def segment_signal(x, segment_size, stride):
    x = np.asarray(x, dtype=np.float32)
    windows = []
    for start in range(0, len(x) - segment_size + 1, stride):
        windows.append(x[start:start + segment_size])
    return windows
 
 
def build_psd_dataset(df_signals, segment_size, stride, fs, nperseg, noverlap):
    rows = []
 
    for _, row in df_signals.iterrows():
        segments = segment_signal(row["signal"], segment_size, stride)
 
        for seg_idx, seg in enumerate(segments):
            psd = compute_psd(seg, fs=fs, nperseg=nperseg, noverlap=noverlap)
 
            rows.append({
                "label":       row["label"],
                "signal":      psd,           # PSD array stored here
                "Feature":     row["Feature"],
                "Bandymas":    row["Bandymas"],
                "Apkrova(Nm)": row["Apkrova(Nm)"],
                "Sukiai(rpm)": row["Sukiai(rpm)"],
                "window_id":   seg_idx
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
    test_signals  = grouped[grouped["Bandymas"] == "2 bandymas"].copy()
 
    n_segs = EXPECTED_SIGNAL_LEN // SEGMENT_SIZE
    welch_windows = (SEGMENT_SIZE - NPERSEG) // (NPERSEG - NOVERLAP) + 1
    psd_len = NPERSEG // 2 + 1
    print(f"\nParametrai:")
    print(f"  Segmento ilgis:     {SEGMENT_SIZE}")
    print(f"  Segmentų per signalą: {n_segs}")
    print(f"  Welch langas:       {NPERSEG}, persidengimas: {NOVERLAP}")
    print(f"  Welch langų per segmentą: {welch_windows}")
    print(f"  PSD vektoriaus ilgis: {psd_len}")

    print(f"\nComputing PSD for train segments...")
    train_windows = build_psd_dataset(
        train_signals, SEGMENT_SIZE, STRIDE, SAMPLE_RATE, NPERSEG, NOVERLAP
    )

    print("Computing PSD for test segments...")
    test_windows = build_psd_dataset(
        test_signals, SEGMENT_SIZE, STRIDE, SAMPLE_RATE, NPERSEG, NOVERLAP
    )
 
    example_psd = train_windows.iloc[0]["signal"]
    print(f"\nPSD bins per segment: {len(example_psd)}  (expected {psd_len})")
    print(f"PSD value range:      [{example_psd.min():.2f}, {example_psd.max():.2f}] dB")

    print(f"\nTrain segments: {len(train_windows)}")
    print(f"Test segments:  {len(test_windows)}")

    print("\nTrain class counts:")
    print(train_windows["label"].value_counts().sort_index())
    print("\nTest class counts:")
    print(test_windows["label"].value_counts().sort_index())

    out_dir = os.path.join(OUT_ROOT, PROBLEM)
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, f"{PROBLEM}_TRAIN.ts")
    test_path  = os.path.join(out_dir, f"{PROBLEM}_TEST.ts")


    print("\nWriting .ts files...")
    write_ts_univariate(train_windows[["signal", "label"]], train_path, PROBLEM)
    write_ts_univariate(test_windows[["signal", "label"]],  test_path,  PROBLEM)

    print("\nSaved:")
    print(f"  {train_path}")
    print(f"  {test_path}")
 
if __name__ == "__main__":
    main()