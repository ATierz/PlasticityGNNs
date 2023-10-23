from src.dataLoader.loader import DataLoader
from pathlib import Path
import argparse


if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Abaqus Geometry Modifier')
    parser.add_argument('--path_to_data', type=str)
    args = parser.parse_args()  # Parse command-line arguments

    # Read Data from .txt files
    dataloader = DataLoader(args.path_to_data)
    dataset = dataloader.load(desired_simulations=None)
    dataloader.save(Path(args.path_to_data) / 'dataset_df.csv')



