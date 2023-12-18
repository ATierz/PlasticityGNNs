from src.dataLoader.dataset import GraphDataset
from src.constants import TRAIN_SIMULATIONS, TEST_SIMULATIONS
from src.plots.plots import plot_graph_data
import argparse
import os
from PIL import Image
from pathlib import Path

STATE_VARIABLES = ['U.Magnitude', 'U.U1', 'U.U2', 'S.Mises', 'COORD.COOR1', 'COORD.COOR2']

if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Abaqus Geometry Modifier')
    parser.add_argument('--path_to_data', type=str, default=r'C:\Users\AMB\Documents\PhD\code\PlasticityGNNs\outputs')
    parser.add_argument('--save_csv', type=bool, default=False)
    args = parser.parse_args()  # Parse command-line arguments

    # Build dataset
    print('\nPreparing TRAIN dataset...')
    train_graph_dataset = GraphDataset(args.path_to_data, sim_type='Beam',
                                       simulation_names=os.listdir(args.path_to_data), state_variables=STATE_VARIABLES,
                                       dataset_name='Beam_visco_2D_05_test', train_flag=False, save_csv=args.save_csv,
                                       desired_regions={'GLASS-1.Region_1': 0, 'GLASS-1.Region_4': 0, 'LIQUID-1.Region_2': 1})
                                       # desired_regions={'GLASS-1.Region_1': 0, 'GLASS-1.Region_4': 0, 'LIQUID-1.Region_2': 1})
    # print(f'Train dataset size: {len(train_graph_dataset)}')

    # print('\nPreparing TEST dataset...')
    # test_graph_dataset = GraphDataset(args.path_to_data,
    #                                    simulation_names=TEST_SIMULATIONS,
    #                                    state_variables=STATE_VARIABLES)
    # print(f'Test dataset size: {len(test_graph_dataset)}')

    # get dataloader
    # train_dataloader = train_graph_dataset.get_loader(batch_size=1)
    # test_dataloader = test_graph_dataset.get_loader(batch_size=32)
