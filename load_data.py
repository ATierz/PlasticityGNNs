from src.dataLoader.loader import DataLoader
from pathlib import Path
import argparse
import json


if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Abaqus Geometry Modifier')
    parser.add_argument('--path_to_data', type=str)
    args = parser.parse_args()  # Parse command-line arguments

    # Read Data from .txt files
    dataloader = DataLoader(args.path_to_data)

    data_nodal_variables = dataloader.load_nodal_variables(desired_simulations=None)
    dataloader.save(Path(args.path_to_data) / 'data_nodal_variables.csv')

    data_edges = dataloader.load_edges(desired_simulations=None)





