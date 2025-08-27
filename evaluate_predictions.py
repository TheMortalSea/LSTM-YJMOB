import os
import argparse
import csv
import gzip
import numpy as np
from collections import defaultdict
from tqdm import tqdm

try:
    import geobleu as gb
except ImportError:
    print("Error: 'geobleu' module not found. Please ensure 'geobleu.py' is in the same directory or installed.")
    exit(1)

def load_trajectories(file_path):
    """
    Loads trajectories from a CSV file (regular or gzip compressed) and groups them by user ID.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        dict: A dictionary where keys are user IDs (uids) and values
              are lists of trajectory points.
    """
    trajectories = defaultdict(list)
    
    try:
        # Check if file is gzipped
        if file_path.endswith('.gz'):
            file_obj = gzip.open(file_path, 'rt', newline='')
        else:
            file_obj = open(file_path, mode='r', newline='')
        
        with file_obj as file:
            reader = csv.reader(file)
            # Skip the header row
            header = next(reader)
            
            # Update header check for new format (without step_id)
            if header != ['uid', 'd', 't', 'x', 'y']:
                raise ValueError(f"CSV file header does not match expected format. Got: {header}")
            
            for row in tqdm(reader, desc=f"Loading {os.path.basename(file_path)}"):
                # Convert string values to the correct type
                user_id, d, t, x, y = map(int, row)
                # Group data by user_id (using user_id as uid for compatibility)
                trajectories[user_id].append((user_id, d, t, x, y))
                
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except ValueError as e:
        print(f"Error parsing file '{file_path}': {e}")
        return None
    
    # Sort each trajectory by day and time for consistency
    for uid in trajectories:
        trajectories[uid].sort(key=lambda p: (p[1], p[2]))
    
    return trajectories

def evaluate_single_city(ground_truth_path, predictions_path, city_name):
    """
    Evaluates predictions for a single city by calculating GeoBLEU and DTW scores.
    Args:
        ground_truth_path (str): Path to the ground truth CSV file.
        predictions_path (str): Path to the predictions CSV file.
        city_name (str): Name of the city being evaluated.
    Returns:
        dict: Dictionary containing evaluation results.
    """
    print(f"\n=== Evaluating City {city_name} ===")
    
    # Load data from the two CSV files
    ground_truth_data = load_trajectories(ground_truth_path)
    if not ground_truth_data:
        return None
        
    predictions_data = load_trajectories(predictions_path)
    if not predictions_data:
        return None
    
    # Find common users between the two datasets
    common_uids = sorted(list(set(ground_truth_data.keys()) & set(predictions_data.keys())))
    
    if not common_uids:
        print("No common users found between the ground truth and prediction files.")
        return None
    
    print(f"Found {len(common_uids)} common users to evaluate.")
    
    geobleu_scores = []
    dtw_scores = []
    trajectory_lengths = []
    
    # Iterate through each user and calculate metrics
    for uid in tqdm(common_uids, desc=f"Calculating metrics for City {city_name}"):
        ans_seq = ground_truth_data[uid]
        sys_seq = predictions_data[uid]
        
        try:
            # Calculate GeoBLEU score for the current user
            geobleu_score = gb.calc_geobleu_single(sys_seq, ans_seq)
            geobleu_scores.append(geobleu_score)
            
            # Calculate DTW score for the current user
            dtw_score = gb.calc_dtw_single(sys_seq, ans_seq)
            dtw_scores.append(dtw_score)
            
            # Track trajectory length
            trajectory_lengths.append(len(ans_seq))
            
        except ValueError as e:
            print(f"Skipping user {uid} due to a data error: {e}")
            continue
    
    if not geobleu_scores or not dtw_scores:
        print(f"No valid scores calculated for City {city_name}")
        return None
    
    # Calculate the average scores across all users
    avg_geobleu = np.mean(geobleu_scores)
    avg_dtw = np.mean(dtw_scores)
    avg_traj_length = np.mean(trajectory_lengths)
    
    # Calculate weighted averages (by trajectory length)
    weights = np.array(trajectory_lengths)
    weighted_avg_geobleu = np.average(geobleu_scores, weights=weights)
    weighted_avg_dtw = np.average(dtw_scores, weights=weights)
    
    print(f"\n--- City {city_name} Results ---")
    print(f"Number of users: {len(geobleu_scores)}")
    print(f"Average trajectory length: {avg_traj_length:.1f}")
    print(f"Average GeoBLEU Score: {avg_geobleu:.4f}")
    print(f"Weighted GeoBLEU Score: {weighted_avg_geobleu:.4f}")
    print(f"Average DTW Score: {avg_dtw:.4f}")
    print(f"Weighted DTW Score: {weighted_avg_dtw:.4f}")
    print("--------------------------------")
    
    return {
        'city': city_name,
        'num_users': len(geobleu_scores),
        'avg_trajectory_length': avg_traj_length,
        'avg_geobleu': avg_geobleu,
        'weighted_geobleu': weighted_avg_geobleu,
        'avg_dtw': avg_dtw,
        'weighted_dtw': weighted_avg_dtw,
        'individual_geobleu': geobleu_scores,
        'individual_dtw': dtw_scores,
        'trajectory_lengths': trajectory_lengths
    }

def evaluate_all_cities(save_path, cities):
    """
    Evaluates all specified cities and provides summary statistics.
    Args:
        save_path (str): Directory containing the generated and reference CSV files.
        cities (list): List of city letters to evaluate.
    """
    all_results = []
    all_geobleu_scores = []
    all_dtw_scores = []
    all_trajectory_lengths = []
    
    # Process each city
    for city in cities:
        # Construct file paths with city_<letter> naming convention
        reference_file = os.path.join(save_path, f'city_{city}_reference.csv.gz')
        generated_file = os.path.join(save_path, f'city_{city}_test_generated.csv.gz')
        
        # Check if both files exist
        if not os.path.exists(reference_file):
            print(f"Warning: Reference file for city {city} not found: {reference_file}")
            continue
        if not os.path.exists(generated_file):
            print(f"Warning: Generated file for city {city} not found: {generated_file}")
            continue
        
        # Evaluate this city
        city_result = evaluate_single_city(reference_file, generated_file, city)
        
        if city_result is not None:
            all_results.append(city_result)
            # Collect all individual scores for overall statistics
            all_geobleu_scores.extend(city_result['individual_geobleu'])
            all_dtw_scores.extend(city_result['individual_dtw'])
            all_trajectory_lengths.extend(city_result['trajectory_lengths'])
    
    # Print overall summary
    if all_results:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY ACROSS ALL CITIES")
        print(f"{'='*60}")
        
        total_users = sum(r['num_users'] for r in all_results)
        overall_avg_geobleu = np.mean(all_geobleu_scores)
        overall_avg_dtw = np.mean(all_dtw_scores)
        overall_avg_traj_length = np.mean(all_trajectory_lengths)
        
        # Weighted averages across all users
        weights = np.array(all_trajectory_lengths)
        overall_weighted_geobleu = np.average(all_geobleu_scores, weights=weights)
        overall_weighted_dtw = np.average(all_dtw_scores, weights=weights)
        
        print(f"Cities evaluated: {[r['city'] for r in all_results]}")
        print(f"Total users across all cities: {total_users}")
        print(f"Overall average trajectory length: {overall_avg_traj_length:.1f}")
        print(f"\nOverall GeoBLEU Score: {overall_avg_geobleu:.4f}")
        print(f"Overall Weighted GeoBLEU: {overall_weighted_geobleu:.4f}")
        print(f"Overall DTW Score: {overall_avg_dtw:.4f}")
        print(f"Overall Weighted DTW: {overall_weighted_dtw:.4f}")
        
        print(f"\nPer-City Summary:")
        for result in all_results:
            print(f"  City {result['city']}: Users={result['num_users']}, "
                  f"GeoBLEU={result['avg_geobleu']:.4f}, DTW={result['avg_dtw']:.4f}")
        
        print(f"{'='*60}")
    else:
        print("No cities were successfully evaluated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trajectory prediction models for multiple cities.")
    parser.add_argument('--save_path', type=str, default='/content/drive/MyDrive',
                       help="Directory containing the generated and reference CSV files.")
    parser.add_argument('--cities', nargs='*', type=str, default=['B', 'C', 'D'],
                       help='Cities to evaluate (e.g., --cities B C D). Defaults to B C D')
    
    args = parser.parse_args()
    
    print(f"Save path: {args.save_path}")
    print(f"Cities to evaluate: {args.cities}")
    
    evaluate_all_cities(args.save_path, args.cities)