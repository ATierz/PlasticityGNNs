import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.dataLoader.builder import DataBuilder
from src.constants import STATE_VARIABLES


class GraphDataset(Dataset):
    def __init__(self, path_to_data, simulation_names=None,
                 state_variables=STATE_VARIABLES):
        """
        Initialize the CustomDataset.

        Args:
            path_to_data (str): The directory containing your dataset.
            simulation_names (str, optional): simulation names of interest to load.
        """

        # Load data from generated files from Abaqus
        databuilder = DataBuilder(path_to_data)
        # Load the nodal variables and edges data
        print(f'Building data from simulations --> {simulation_names}')
        self.data_nodal_variables = databuilder.get_nodal_variables(desired_simulations=simulation_names)