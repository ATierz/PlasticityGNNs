import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.dataLoader.builder import DataBuilder
from src.constants import STATE_VARIABLES


class GraphDataset(Dataset):
    def __init__(self, path_to_data, sim_type, simulation_names=None,
                 state_variables=STATE_VARIABLES, dataset_name='', train_flag=True, save_csv=False,
                 desired_regions='all'):
        """
        Initialize the CustomDataset.

        Args:
            path_to_data (str): The directory containing your dataset.
            simulation_names (str, optional): simulation names of interest to load.
        """

        # Load data from generated files from Abaqus
        databuilder = DataBuilder(path_to_data, dataset_name, desired_regions=desired_regions)
        # Load the nodal variables and edges data
        print(f'Building data from simulations --> {simulation_names}')
        databuilder.get_nodal_variables(sim_type, desired_simulations=simulation_names, save_csv=save_csv)
        if sim_type == 'Glass':
            self.data_total = databuilder.calculate_glass_dataset(train_flag=train_flag)
        elif sim_type == 'Beam3D':
            self.data_total = databuilder.calculate_beam_dataset_3D()
        elif sim_type == 'Beam':
            self.data_total = databuilder.calculate_beam_dataset(train_flag=train_flag)
        # self.data_edges = databuilder.get_edges(desired_simulations=simulation_names)
        # save dataset in .pt format
        databuilder.save_dataset(self.data_total, simulation_names=simulation_names, train_flag=train_flag)
        # set state variables of interest
        # self.state_variables = state_variables
        # # select simulations to load
        # self.simulation_names = [simulation_name for simulation_name, _ in
        #                          self.data_edges.items()] if simulation_names is None else simulation_names
        # # get data as a list into Torch Geometric data structure
        # self.data = self.__build_batch_data(self.simulation_names)

    def __build_batch_data(self, simulations_list):

        batch_data = []
        for simulation_name in simulations_list:

            edge_index = torch.tensor(self.data_edges[simulation_name]['edge_index'], dtype=torch.long)
            simulation_data_nodal_variables = self.data_nodal_variables[
                self.data_nodal_variables['Simulation'] == simulation_name]
            max_frame_increment = self.data_nodal_variables[self.data_nodal_variables['Simulation'] == simulation_name][
                'Frame_increment'].max()

            for frame_increment in range(max_frame_increment - 1):
                z_t = torch.tensor(simulation_data_nodal_variables[
                                       simulation_data_nodal_variables['Frame_increment'] == frame_increment][
                                       self.state_variables].values, dtype=torch.float)
                z_t_1 = torch.tensor(simulation_data_nodal_variables[
                                         simulation_data_nodal_variables['Frame_increment'] == frame_increment + 1][
                                         self.state_variables].values, dtype=torch.float)

                data_geometric = Data(x=z_t, edge_index=edge_index, y=z_t_1)
                batch_data.append(data_geometric)

        return batch_data

    def __len__(self):
        return len(self.data)

    def get_loader(self, batch_size=32, shuffle=False):
        return DataLoader(self.data, batch_size=batch_size, shuffle=shuffle)
