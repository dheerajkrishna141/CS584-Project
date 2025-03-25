import pandas as pd
import glob
import os

def load_and_label_signals():
    """Load all optimized_results_*.csv files and label them by date."""
    files = sorted(glob.glob("optimized_results_*.csv"))

    if not files:
        print("No optimized_results_*.csv files found.")
        return None, None

    all_data = []
    file_labels = []

    for f in files:
        try:
            df = pd.read_csv(f, usecols=["Symbol", "Last_Signal"])
            label = os.path.splitext(os.path.basename(f))[0].replace("optimized_results_", "")
            df.rename(columns={"Last_Signal": label}, inplace=True)
            all_data.append(df)
            file_labels.append(label)
        except Exception as e:
            print(f"Failed to process {f}: {e}")

    return all_data, file_labels

def compare_signals():
    dataframes, labels = load_and_label_signals()

    if not dataframes:
        return

    # Merge all dataframes on Symbol
    df_merged = dataframes[0]
    for df in dataframes[1:]:
        df_merged = pd.merge(df_merged, df, on="Symbol", how="outer")

    df_merged.sort_values("Symbol", inplace=True)

    # Detect changes in signals
    def signal_changed(row):
        signals = row[1:]  # exclude Symbol
        unique_signals = set(signal for signal in signals if pd.notna(signal))
        return "✅ Yes" if len(unique_signals) > 1 else "-"

    df_merged["Changed"] = df_merged.apply(signal_changed, axis=1)

    print("\nSignal Comparison Table (Flips Highlighted):")
    print(df_merged.fillna("-").to_string(index=False))

    # Save results to CSV
    df_merged.to_csv("last_signal_comparison.csv", index=False)
    print("\nComparison saved to last_signal_comparison.csv")

if __name__ == "__main__":
    compare_signals()
