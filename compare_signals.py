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

    if not dataframes or len(dataframes) < 2:
        print("Need at least two files to compare signals.")
        return

    # Merge all dataframes on Symbol
    df_merged = dataframes[0]
    for df in dataframes[1:]:
        df_merged = pd.merge(df_merged, df, on="Symbol", how="outer")

    df_merged.sort_values("Symbol", inplace=True)

    # Only compare the last two columns (most recent signals)
    last_two_labels = labels[-2:]  # e.g., ['032624', '032724']

    def signal_changed(row):
        prev, curr = row[last_two_labels[0]], row[last_two_labels[1]]
        if pd.isna(prev) or pd.isna(curr):
            return "-"
        return "✅ Yes" if prev != curr else "-"

    df_merged["Changed"] = df_merged.apply(signal_changed, axis=1)

    print(f"\nSignal Comparison Table (Comparing {last_two_labels[0]} → {last_two_labels[1]}):")
    print(df_merged.fillna("-").to_string(index=False))

    df_merged.to_csv("last_signal_comparison.csv", index=False)
    print("\nComparison saved to last_signal_comparison.csv")

if __name__ == "__main__":
    compare_signals()
