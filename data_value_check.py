import pandas as pd
import numpy as np
import argparse

def inspect_test_data(path):
    """
    Loads a dataset file, identifies the test set users, and prints
    the unique values for x and y coordinates.
    """
    try:
        # Read the dataset file
        print(f"Loading data from: {path}")
        traj_df = pd.read_csv(path, compression='gzip')

        # Identify the test set users based on the logic in your script
        # This assumes city B, C, or D where the last 3000 users are the test set
        test_df = traj_df[traj_df['uid'] >= len(pd.unique(traj_df['uid'])) - 3000]

        if test_df.empty:
            print("No test data found. This script is intended for cities B, C, or D.")
            return

        print("\n--- Unique X Coordinates in the Test Set ---")
        unique_x = np.unique(test_df['x'].to_numpy())
        print(unique_x)

        print("\n--- Unique Y Coordinates in the Test Set ---")
        unique_y = np.unique(test_df['y'].to_numpy())
        print(unique_y)

        # Confirm the presence of the mask value
        if 999 in unique_x or 999 in unique_y:
            print("\nAnalysis: The value 999 is present in the coordinates. This confirms the raw data contains the mask value.")
            print("Your model's embedding layers are not designed for an index of 999, which is why the filtering step is necessary.")
        else:
            print("\nAnalysis: The value 999 was not found. The issue may be with other invalid data points or a different part of the code.")

    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found. Please check your file path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspects raw test data for unexpected coordinate values.")
    parser.add_argument('--path', type=str, required=True,
                        help='The path to the dataset file (e.g., ./dataset/city_B_challengedata.csv.gz)')
    args = parser.parse_args()
    inspect_test_data(args.path)
