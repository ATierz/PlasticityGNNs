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
        self.data_edges = databuilder.get_edges(desired_simulations=simulation_names)
        # set state variables of interest
        self.state_variables = state_variables
        # select simulations to load
        self.simulation_names = [simulation_name for simulation_name, _ in self.data_edges.items()] if simulation_names is None else simulation_names
        # get data as a list into Torch Geometric data structure
        self.data = self.__build_batch_data(self.simulation_names)

    def __build_batch_data(self, simulations_list):
        batch_data = []
        for simulation_name in simulations_list:

            edge_index = torch.tensor(self.data_edges[simulation_name]['edge_index'], dtype=torch.long)

            simulation_data_nodal_variables = self.data_nodal_variables[self.data_nodal_variables['Simulation'] == simulation_name]

            max_frame_increment = self.data_nodal_variables[self.data_nodal_variables['Simulation'] == simulation_name]['Frame_increment'].max()

            for frame_increment in range(max_frame_increment - 1):

                z_t = torch.tensor(simulation_data_nodal_variables[simulation_data_nodal_variables['Frame_increment']==frame_increment][self.state_variables].values, dtype=torch.float)
                z_t_1 = torch.tensor(simulation_data_nodal_variables[simulation_data_nodal_variables['Frame_increment']==frame_increment + 1][self.state_variables].values, dtype=torch.float)

                data_geometric = Data(x=z_t, edge_index=edge_index, y=z_t_1)
                batch_data.append(data_geometric)

        return batch_data

    def __len__(self):
        return len(self.data)

    def get_loader(self, batch_size=32, shuffle=False):
        return DataLoader(self.data, batch_size=batch_size, shuffle=shuffle)