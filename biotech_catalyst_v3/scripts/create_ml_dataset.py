"""Combine high-move and low-move Clinical Data for ML."""

import pandas as pd
import argparse


def create_ml_dataset(
    high_file: str = "enriched_high_moves.csv",
    low_file: str = "low_move_enriched.csv",
    output_file: str = "ml_dataset_clinical.csv",
):
    # Load datasets
    high = pd.read_csv(high_file)
    low = pd.read_csv(low_file)

    print(f"High-move data: {len(high)} rows")
    print(f"Low-move data: {len(low)} rows")

    # Filter to Clinical Data only
    high = high[high['catalyst_type'] == 'Clinical Data'].copy()
    low = low[low['catalyst_type'] == 'Clinical Data'].copy()

    print(f"High-move Clinical Data: {len(high)}")
    print(f"Low-move Clinical Data: {len(low)}")

    # Add target labels
    high['move_class'] = 'High'  # normalized_move > 3.0 (or raw > 30%)
    low['move_class'] = 'Low'    # normalized_move < 1.5

    # Balance: take min of both
    n = min(len(high), len(low))
    high_balanced = high.sample(n=n, random_state=42)
    low_balanced = low.sample(n=n, random_state=42)

    # Combine
    ml_data = pd.concat([high_balanced, low_balanced], ignore_index=True)
    ml_data = ml_data.sample(frac=1, random_state=42)  # Shuffle

    # Save
    ml_data.to_csv(output_file, index=False)

    print(f"\nML Dataset: {len(ml_data)} events")
    print(f"  High-move: {(ml_data['move_class'] == 'High').sum()}")
    print(f"  Low-move: {(ml_data['move_class'] == 'Low').sum()}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create balanced ML dataset")
    parser.add_argument("--high", default="enriched_high_moves.csv")
    parser.add_argument("--low", default="low_move_enriched.csv")
    parser.add_argument("--output", default="ml_dataset_clinical.csv")
    args = parser.parse_args()

    create_ml_dataset(
        high_file=args.high,
        low_file=args.low,
        output_file=args.output,
    )
