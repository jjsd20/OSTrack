import os
import json
from typing import Dict, List, Tuple

global kf
global nkf


def parse_evaluation_file(file_path: str) -> Dict[str, float]:
    """Parse an evaluation file and extract metrics."""
    results = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'Average Overlap':
                        results[key] = float(value)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return results


def compare_folders(folder1: str, folder2: str) -> List[Tuple[str, float, float, float]]:
    """Compare evaluation files between two folders and calculate differences."""
    differences = []
    kf=0
    n_kf=0

    # Get all evaluation files in folder1
    for filename in os.listdir(folder1):
        if filename.endswith('_evaluation.txt'):
            file1 = os.path.join(folder1, filename)
            file2 = os.path.join(folder2, filename)

            if os.path.exists(file2):
                # Parse both files
                result1 = parse_evaluation_file(file1)
                result2 = parse_evaluation_file(file2)

                if 'Average Overlap' in result1 and 'Average Overlap' in result2:
                    overlap1 = result1['Average Overlap']
                    overlap2 = result2['Average Overlap']
                    diff = (overlap1 - overlap2)
                    if diff >=0:
                        kf = kf + 1
                    else: n_kf = n_kf + 1

                    # Extract video name (assuming format "videoName-XX_evaluation.txt")
                    video_name = filename.split('_')[0]
                    differences.append((video_name, overlap1, overlap2, diff))

    # Sort by difference in descending order
    differences.sort(key=lambda x: x[3], reverse=True)
    return differences, kf, n_kf


def save_results_to_json(results: List[Tuple[str, float, float, float]], output_path: str):
    """Save comparison results to a JSON file."""
    json_data = []
    for video, overlap1, overlap2, diff in results:
        json_data.append({
            'video': video,
            'average_overlap_1': overlap1,
            'average_overlap_2': overlap2,
            'difference': diff
        })

    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)


if __name__ == '__main__':
    folder1 = '/home/ty/OSTrack/output/test/tracking_results/kmostrack/vitb_256_mae_ce_96x1_ep300-3'
    #folder2 = '/home/ty/OSTrack/output/test/tracking_results/ostrack/vitb_256_mae_ce_96x1_ep300'
    folder2 = '/home/ty/OSTrack/output/test/tracking_results/kmostrack/vitb_256_mae_ce_96x1_ep300-2'
    output_file = 'e300-3-2-results.json'

    results,kf,nkf = compare_folders(folder1, folder2)
    save_results_to_json(results, output_file)
    print(f"Results saved to {output_file}")
    print(f"KF: {kf}")
    print(f"NKF: {nkf}")

