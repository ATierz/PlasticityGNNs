from src.dataLoader.dataset import GraphDataset
from src.constants import TRAIN_SIMULATIONS, TEST_SIMULATIONS
from src.plots.plots import plot_graph_data
import argparse

from PIL import Image
from pathlib import Path


STATE_VARIABLES = ['U.Magnitude', 'U.U1', 'U.U2', 'S.Mises', 'COORD.COOR1', 'COORD.COOR2']

if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Abaqus Geometry Modifier')
    parser.add_argument('--path_to_data', type=str)
    args = parser.parse_args()  # Parse command-line arguments

    # Build dataset
    print('\nPreparing TRAIN dataset...')
    train_graph_dataset = GraphDataset(args.path_to_data,
                                 simulation_names=['rectangle_L1.0_R1.0'],
                                 state_variables=STATE_VARIABLES)
    print(f'Train dataset size: {len(train_graph_dataset)}')

    print('\nPreparing TEST dataset...')
    test_graph_dataset = GraphDataset(args.path_to_data,
                                       simulation_names=TEST_SIMULATIONS,
                                       state_variables=STATE_VARIABLES)
    print(f'Test dataset size: {len(test_graph_dataset)}')

    # get dataloader
    train_dataloader = train_graph_dataset.get_loader(batch_size=1)
    test_dataloader = test_graph_dataset.get_loader(batch_size=32)





