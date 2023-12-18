from src.dataLoader.dataset import GraphDataset
from src.constants import TRAIN_SIMULATIONS, TEST_SIMULATIONS
from src.plots.plots import make_gif
import argparse

from pathlib import Path


STATE_VARIABLES = ['U.Magnitude', 'U.U1', 'U.U2', 'S.Mises', 'COORD.COOR1', 'COORD.COOR2']

if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Abaqus Geometry Modifier')
    parser.add_argument('--path_to_data', type=str, default=r'C:\Users\AMB\Documents\PhD\code\PlasticityGNNs\outputs')
    args = parser.parse_args()  # Parse command-line arguments
    simulation_name = 'Foam_800_200_450_25_0012'
    # for simulation_name in TRAIN_SIMULATIONS + TEST_SIMULATIONS:
        # Build dataset
    print('\nPreparing TRAIN dataset...')
    train_graph_dataset = GraphDataset(args.path_to_data,
                                 simulation_names=[simulation_name],
                                 state_variables=STATE_VARIABLES)
    print(f'Train dataset size: {len(train_graph_dataset)}')

    # get dataloader
    train_dataloader = train_graph_dataset.get_loader(batch_size=1)

    data = [sample for sample in train_dataloader]

    output = Path(args.path_to_data) / simulation_name / 'gifs'
    output.mkdir(exist_ok=True)

    make_gif(data, output, f'{simulation_name} S.Mises')






