# import pandas as pd
# import os
# import argparse
# import sys

# def compress_validation_data(input_paths, output_dir):
#     """
#     Reads multiple uncompressed CSV files, typically containing validation data,
#     subsets the data to the last 3000 unique users,
#     and saves them as compressed gzip files in a specified directory.
    
#     Args:
#         input_paths (list): A list of paths to the input .csv files.
#         output_dir (str): The directory where the compressed .csv.gz files will be saved.
#     """
#     # Create the output directory if it does not exist
#     if not os.path.exists(output_dir):
#         print(f"Creating output directory: {output_dir}")
#         os.makedirs(output_dir)

#     for input_path in input_paths:
#         try:
#             # Check if the file exists and is a CSV
#             if not os.path.exists(input_path) or not input_path.endswith('.csv'):
#                 print(f"Skipping '{input_path}': Not a valid CSV file or does not exist.")
#                 continue

#             # Read the uncompressed CSV file using pandas
#             print(f"Reading data from: {input_path}")
#             df = pd.read_csv(input_path)

#             # --- Subsetting the data to the last 3000 UIDs ---
#             print(f"Subsetting data to the last 3000 unique users...")
            
#             # Get the unique user IDs
#             all_uids = df['uid'].unique()
            
#             # Check if there are at least 3000 users before trying to subset
#             if len(all_uids) < 3000:
#                 print(f"Warning: The dataset at '{input_path}' contains fewer than 3000 unique users. Skipping subsetting.")
#             else:
#                 # Get the UIDs for the last 3000 users
#                 test_uids = all_uids[-3000:]
                
#                 # Filter the DataFrame to only include these test set users
#                 df = df[df['uid'].isin(test_uids)].copy()

#             # Get the base filename and construct the new gzipped name
#             base_name = os.path.basename(input_path)
#             output_name = base_name.replace('.csv', '.csv.gz')
#             output_path = os.path.join(output_dir, output_name)

#             # Save the DataFrame to a compressed CSV file
#             print(f"Writing compressed data to: {output_path}")
#             df.to_csv(output_path, index=False, compression='gzip')

#             print(f"Successfully compressed '{base_name}' to '{output_name}'")

#         except Exception as e:
#             print(f"An error occurred while processing '{input_path}': {e}")

# if __name__ == '__main__':
#     # Set up argument parser to handle command-line arguments for flexibility
#     parser = argparse.ArgumentParser(description="Compress multiple CSV files to gzip format.")
#     parser.add_argument('--cities', nargs='+', type=str,
#                         help='A list of cities to process (e.g., --cities B C D).',
#                         default=['B', 'C', 'D'])
#     parser.add_argument('--input_dir', type=str, default='.',
#                         help='The directory containing the uncompressed .csv files.')
#     parser.add_argument('--output_dir', type=str, default='.',
#                         help='The directory to save the compressed .csv.gz files.')

#     # Parse the arguments from the command line
#     args = parser.parse_args()

#     # Construct the input file paths based on the specified cities
#     input_file_paths = []
#     for city in args.cities:
#         input_file_paths.append(os.path.join(args.input_dir, f'city{city}-dataset.csv'))
    
#     # Call the main compression function
#     compress_validation_data(input_file_paths, args.output_dir)

import pandas as pd
import os
import argparse
import sys

def compress_validation_data(input_paths, output_dir):
    """
    Reads multiple uncompressed CSV files, typically containing validation data,
    subsets the data to the last 3000 unique users,
    and saves them as compressed gzip files in a specified directory.
    
    Args:
        input_paths (list): A list of paths to the input .csv files.
        output_dir (str): The directory where the compressed .csv.gz files will be saved.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    for input_path in input_paths:
        try:
            # Check if the file exists and is a CSV
            if not os.path.exists(input_path) or not input_path.endswith('.csv'):
                print(f"Skipping '{input_path}': Not a valid CSV file or does not exist.")
                continue

            # Read the uncompressed CSV file using pandas
            print(f"Reading data from: {input_path}")
            df = pd.read_csv(input_path)

            # Get the unique user IDs from the full dataset
            all_uids = df['uid'].unique()
            
            # --- Print the UID ranges of the full dataset ---
            if len(all_uids) > 0:
                print(f"Full Dataset UID Range:")
                print(f"  Total unique UIDs: {len(all_uids)}")
                print(f"  UIDs from {min(all_uids)} to {max(all_uids)}")
            else:
                print("Warning: The dataset is empty or has no UIDs.")
            
            # --- Subsetting the data to the last 3000 UIDs ---
            print(f"\nSubsetting data to the last 3000 unique users...")
            
            # Check if there are at least 3000 users before trying to subset
            if len(all_uids) < 3000:
                print(f"Warning: The dataset at '{input_path}' contains fewer than 3000 unique users. Skipping subsetting.")
                test_uids = all_uids # Keep all UIDs if the total is less than 3000
            else:
                # Get the UIDs for the last 3000 users
                test_uids = all_uids[-3000:]
                
                # Filter the DataFrame to only include these test set users
                df = df[df['uid'].isin(test_uids)].copy()

            # --- Print the UID ranges of the validation set ---
            if len(test_uids) > 0:
                print(f"Validation Dataset UID Range:")
                print(f"  Total unique UIDs: {len(test_uids)}")
                print(f"  UIDs from {min(test_uids)} to {max(test_uids)}")
            else:
                print("Warning: The validation dataset is empty.")

            # Get the base filename and construct the new gzipped name
            base_name = os.path.basename(input_path)
            output_name = base_name.replace('.csv', '.csv.gz')
            output_path = os.path.join(output_dir, output_name)

            # Save the DataFrame to a compressed CSV file
            print(f"\nWriting compressed data to: {output_path}")
            df.to_csv(output_path, index=False, compression='gzip')

            print(f"Successfully compressed '{base_name}' to '{output_name}'")

        except Exception as e:
            print(f"An error occurred while processing '{input_path}': {e}")
            
        print("-" * 50) # Separator for clarity between files

if __name__ == '__main__':
    # Set up argument parser to handle command-line arguments for flexibility
    parser = argparse.ArgumentParser(description="Compress multiple CSV files to gzip format.")
    parser.add_argument('--cities', nargs='+', type=str,
                        help='A list of cities to process (e.g., --cities B C D).',
                        default=['B', 'C', 'D'])
    parser.add_argument('--input_dir', type=str, default='.',
                        help='The directory containing the uncompressed .csv files.')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='The directory to save the compressed .csv.gz files.')

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Construct the input file paths based on the specified cities
    input_file_paths = []
    for city in args.cities:
        input_file_paths.append(os.path.join(args.input_dir, f'city{city}-dataset.csv'))
    
    # Call the main compression function
    compress_validation_data(input_file_paths, args.output_dir)