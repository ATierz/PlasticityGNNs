import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import datetime
import random
from scipy.signal import savgol_filter
from src.dataLoader.reader import DataReader
from src.utils.utils import  compute_connectivity
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# Import necessary libraries, modules, and classes (e.g., Path, DataReader, pandas) should be imported here.

class DataBuilder(object):
    """
    Class with a load method for loading data from ".txt" files within subfolders of a specified data folder.
    It allows you to specify desired simulations and reads the data using the DataReader class. The results are
    stored in a pandas DataFrame, and a 'Simulation' column is added to indicate the simulation name.
    Finally, the individual DataFrames are concatenated into a single dataset DataFrame.
    """
    def __init__(self, path_to_data_folder, dataset_name, train_split=0.9, desired_regions='all'):
        # Constructor to initialize the DataLoader with the path to the data folder
        self.path_to_data_folder = Path(path_to_data_folder)
        self.dataset_df = None  # Initialize a variable to store the final dataset DataFrame
        self.dataset_name = dataset_name
        self.train_split = train_split
        self.desired_regions = desired_regions

    def get_nodal_variables(self, sim_type, desired_simulations=None, save_csv=False):
        # Method to load nodal variable data from text files

        print('Saving nodal variables data...')

        # Initialize an empty list to store DataFrames for each simulation
        dataset_df = []

        # Loop through all ".txt" files within subfolders of the data folder
        for file in self.path_to_data_folder.rglob('**/*.txt'):

            # Extract the simulation name from the file path
            simulation_name = file.parts[-2]
            print(simulation_name)
            csv_path = os.path.join(os.path.dirname(file), f'{simulation_name}.csv')

            if save_csv or not os.path.isfile(csv_path):

                # Check if the simulation is in the list of desired simulations
                if desired_simulations is not None:
                    if simulation_name not in desired_simulations:
                        continue  # Skip this simulation if not desired

                # Use the DataReader to read the data from the text file into a DataFrame
                simulation_df = DataReader(sim_type).get_df_from_txt(file)

                # Insert a new column 'Simulation' with the simulation name
                simulation_df.insert(0, 'Simulation', simulation_name)

                # Append the simulation DataFrame to the dataset list

                simulation_df.to_csv(csv_path, index_label=False)
            # else:
            #     simulation_df = pd.read_csv(csv_path)

            # dataset_df.append(simulation_df)

        # Concatenate the list of DataFrames into a single DataFrame
        # self.dataset_df = pd.concat(dataset_df) if len(dataset_df) > 0 else print('Empty DataFrame, no data found to concatenate.')

        print('Done!')

        # return self.dataset_df  # Return the final dataset DataFrame

    def get_edges(self, data_nodal_variables,desired_simulations=None):
        # Method to load edges data from text files

        print('Loading edges data...')

        # Initialize an empty list to store DataFrames for each simulation
        edges = {}

        # Loop through all ".inp" files within subfolders of the data folder
        for file in self.path_to_data_folder.rglob('**/*.inp'):

            # Extract the simulation name from the file path
            simulation_name = file.parts[-2]

            # Check if the simulation is in the list of desired simulations
            if desired_simulations is not None:
                if simulation_name not in desired_simulations:
                    continue  # Skip this simulation if not desired

            # Use the DataReader to read the data from the text file into a DataFrame
            simulation_dict = DataReader().get_edges_from_inp(file)

            # Append the simulation DataFrame to the dataset list
            edges[simulation_name] = simulation_dict

        print('Done!')

        return edges  # Return the list of loaded edges data

    def reagrupate(self, row, velx, vely):
        pos = row[:, 6:8].astype(float)
        pos[:, 0] = pos[:, 0] - min(pos[:, 0])

        vel_x = velx.reshape(self.n_particles, 1)
        vel_y = vely.reshape(self.n_particles, 1)

        sigma1 = row[:, -4:-2].astype(float)
        sigma2 = row[:, -1].astype(float).reshape(self.n_particles, 1)

        x = np.concatenate((pos, vel_x, vel_y, sigma1, sigma2), axis=1)
        return x

    def calculate_beam_dataset(self, train_flag=True):

        print('Loading edges data...')
        self.start_step = 4
        self.radius_connectivity = 9
        self.dimensions = {'h': 200, 'w': 800, 'mesh': 25, 'f': {'p': 450, 'd': 40}}
        self.sampling_factor = 1

        data_total = []
        for file in self.path_to_data_folder.rglob('**/*.txt'):
            # Extract the simulation name from the file path
            sim = file.parts[-2]
            # print(file)
            csv_path = os.path.join(os.path.dirname(file), f'{sim}.csv')
            data_nodal_variables = pd.read_csv(csv_path)


        # data_total = []
        # for sim in df_region['Simulation'].unique():
            self.dimensions['w'] = int(sim.split('_')[1])
            self.dimensions['h'] = int(sim.split('_')[2])
            self.dimensions['f']['p'] = int(sim.split('_')[3])
            self.dimensions['mesh'] = int(sim.split('_')[4])
            self.dimensions['rows'] = int((self.dimensions['h'] / self.dimensions['mesh']) + 1)
            self.dimensions['columns'] = int((self.dimensions['w'] / self.dimensions['mesh']) + 1)

            ini_f = int((self.dimensions['f']['p'] - (self.dimensions['f']['d'] / 2)))
            fin_f = int((self.dimensions['f']['p'] + (self.dimensions['f']['d'] / 2)))

            # Set up a matrix to represent the mesh
            self.mesh_n = np.ones((self.dimensions['rows'], self.dimensions['columns']))
            # self.mesh_n[:, 0] = 0
            # self.mesh_n[:, -1] = 0  #TODO quitar doble empotramiento
            # ratio_w = self.mesh_n.shape[1] / self.dimensions['w']
            # if sim.split('_')[-1] == 'bott':
            #     self.mesh_n[-1, int((self.dimensions['f']['p'] - (self.dimensions['f']['d'] / 2)) * ratio_w):int((self.dimensions[ 'f'][ 'p'] + (self.dimensions['f']['d'] / 2)) * ratio_w) + 1] = 2
            # else:
            #     self.mesh_n[-1, int((self.dimensions['f']['p'] - (self.dimensions['f']['d'] / 2)) * ratio_w):int((self.dimensions[ 'f'][ 'p'] + (self.dimensions['f']['d'] / 2)) * ratio_w) + 1] = 2

            df = data_nodal_variables.loc[data_nodal_variables['Simulation'] == sim]
            n_steps = len(df['Frame_increment'].unique()) - self.start_step

            n = torch.from_numpy(np.reshape(self.mesh_n, self.mesh_n.shape[0] * self.mesh_n.shape[1]))
            self.n_particles = n.shape[0]


            # Derivate velocities from position
            total_pos = []
            cnt = 0
            for i in range(n_steps):
                step = i + self.start_step
                df_ = df.loc[df['Frame_increment'] == step]
                total_pos.append(np.asarray(df_)[:, 6:8])
                if not len(df_):
                    cnt += 1
            try:
                total_pos = np.asarray(total_pos)
            except:
                print(f'ERRORRRR: {cnt},   {file},')
                continue

            vel_x = []
            vel_y = []
            for particle in range(self.n_particles):
                vel_x.append(savgol_filter(np.gradient(total_pos[:, particle, 0]), 5, 3))
                vel_y.append(savgol_filter(np.gradient(total_pos[:, particle, 1]), 5, 3))
            vel_x = np.asarray(vel_x)
            vel_y = np.asarray(vel_y)

            f_total = []
            xy_total = []

            pos = np.asarray(df.loc[df['Frame_increment'] == 0])[:,6:8].copy()
            n[pos[:, 0] < 0.01] = 0
            if sim.split('_')[-1] == 'bott':
                n[(pos[:, 0] < fin_f+0.1) * (pos[:, 0] > ini_f-0.1) * (pos[:, 1] < 1)] = 2
            else:
                n[(pos[:, 0] < fin_f + 0.1) * (pos[:, 0] > ini_f - 0.1) * (pos[:, 1] > max(pos[:, 1])-1)] = 2
            f = torch.zeros((self.n_particles, 1))
            f[n == 2] = int(sim.split('_')[5]) / 100

            # ramp force
            n_ramp_steps = int((n_steps-1 + self.start_step)/4)
            increment_force = f * (1/n_ramp_steps)
            f = torch.zeros((self.n_particles, 1))

            # for i in range(self.start_step):
            for i in range(1):
                f += increment_force
            # Complet the dataset
            for i in range(n_steps - 1):
                if i < n_ramp_steps-1:
                    f += increment_force
                else:
                    c11=1
                f_total.append(max(f.clone())[0])

                step = i + self.start_step
                # df_x0 = df.loc[df['Frame'] == (step * self.sampling_factor) + 1]
                df_x = df.loc[df['Frame_increment'] == step]
                df_y = df.loc[df['Frame_increment'] == step + 1]

                # y = torch.from_numpy(self.reagrupate(np.asarray(df_y), np.asarray(df_x)[:, 4:6])).to(torch.float32)
                x = torch.from_numpy(self.reagrupate(np.asarray(df_x), vel_x[:, i], vel_y[:, i])).to(torch.float32)
                y = torch.from_numpy(self.reagrupate(np.asarray(df_y), vel_x[:, i + 1], vel_y[:, i + 1])).to(torch.float32)

                pos = x[:, :3].clone()
                pos[:, 2] = pos[:, 2] * 0

                xy_total.append(x[-1, -2].clone())
                newedge_index = compute_connectivity(np.asarray(pos), self.radius_connectivity, add_self_edges=False)
                data_total.append(Data(x, edge_index=newedge_index, y=y, n=n, f=f))
            print()
            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax = fig.add_subplot(3, 1, 1)
            # ax.grid(linestyle='-', linewidth=2)
            # plt.plot(np.asarray(xy_total))
            # ax = fig.add_subplot(3, 1, 2)
            # ax.grid(linestyle='-', linewidth=2)
            # plt.plot(np.asarray(yy_total))
            # ax = fig.add_subplot(3, 1, 3)
            # ax.grid(linestyle='-', linewidth=2)
            # plt.plot(np.asarray(f_total))
            # plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            scatter = ax.scatter(pos[:, 0], pos[:, 1], c=n)
            ax.grid()
            fig.colorbar(scatter, ax=ax, label='Valores')
            plt.savefig('images/'+sim+'png')
            #
            # for i in range(newedge_index.shape[1]):
            #     start = pos[newedge_index[0, i], :2]
            #     end = pos[newedge_index[1, i], :2]
            #     ax.plot([start[0], end[0]], [start[1], end[1]], color='gray')
        if train_flag:
            data_total = random.sample(data_total, k=int(len(data_total) / 1))
        return data_total

    def reagrupate_3D(self, row, velx, vely, velz):
        pos = row[:, 6:9].astype(float)
        pos[:, 0] = pos[:, 0] - min(pos[:, 0])

        vel_x = velx.reshape(self.n_particles, 1)
        vel_y = vely.reshape(self.n_particles, 1)
        vel_z = velz.reshape(self.n_particles, 1)

        sigma = row[:, -6:].astype(float)

        # x = np.concatenate((pos, vel_x, vel_y, vel_z, sigma), axis=1)
        x = np.concatenate((pos[:, 0:1], pos[:, 1:2], pos[:, 2:3], vel_x, vel_y, vel_z, sigma), axis=1)
        # x = np.concatenate((pos[:, 0:1], pos[:, 2:3], pos[:, 1:2], vel_x, vel_z, vel_y, sigma[:, 0:1], sigma[:, 2:3], sigma[:, 1:2], sigma[:, 3:4], sigma[:,5:6], sigma[:,4:5]), axis=1)
        return x
    def calculate_beam_dataset_3D(self):

        print('Loading edges data...')
        self.start_step = 0
        self.radius_connectivity = 55
        self.dimensions = {'h': 200, 'w': 800, 'z':150, 'mesh': 25, 'f': {'p': 450, 'd': 100}}
        self.sampling_factor = 1

        data_total = []
        for file in self.path_to_data_folder.rglob('**/*.txt'):
            # Extract the simulation name from the file path
            sim = file.parts[-2]
            print(file)
            csv_path = os.path.join(os.path.dirname(file), f'{sim}.csv')
            data_nodal_variables = pd.read_csv(csv_path)


        # data_total = []
        # for sim in df_region['Simulation'].unique():
            self.dimensions['w'] = int(sim.split('_')[1])
            self.dimensions['h'] = int(sim.split('_')[2])
            self.dimensions['z'] = int(sim.split('_')[3])
            self.dimensions['f']['p'] = int(sim.split('_')[4])
            self.dimensions['f']['d'] = int(sim.split('_')[3])
            self.dimensions['mesh'] = int(sim.split('_')[5])
            self.dimensions['rows'] = int((self.dimensions['h'] / self.dimensions['mesh']) + 1)
            self.dimensions['columns'] = int((self.dimensions['w'] / self.dimensions['mesh']) + 1)
            self.dimensions['rows_z'] = int((self.dimensions['z'] / self.dimensions['mesh']) + 1)

            # Set up a matrix to represent the mesh
            if sim.split('_')[-1] == 'izq':
                self.mesh_n = np.ones((self.dimensions['rows'], self.dimensions['rows_z'], self.dimensions['columns']))
                # self.mesh_n = np.ones((self.dimensions['rows'], self.dimensions['columns']))
                self.mesh_n[:, :, -1] = 0

                ratio_w = self.mesh_n.shape[2] / self.dimensions['w']
                f_ini = self.dimensions['columns'] - int(
                    (self.dimensions['f']['p'] - (self.dimensions['f']['d'] / 2)) * ratio_w)
                f_fin = self.dimensions['columns'] - int(
                    (self.dimensions['f']['p'] + (self.dimensions['f']['d'] / 2)) * ratio_w)
                self.mesh_n[:, 0, f_fin:f_ini] = 2
            else:
                self.mesh_n = np.ones((self.dimensions['rows'], self.dimensions['columns'], self.dimensions['rows_z']))
                # self.mesh_n = np.ones((self.dimensions['rows'], self.dimensions['columns']))
                self.mesh_n[:, 0, :] = 0
                # self.mesh_n[:, -1, :] = 0  #TODO doble empotramiento
                ratio_w = self.mesh_n.shape[1] / self.dimensions['w']
                self.mesh_n[0, int((self.dimensions['f']['p'] - (self.dimensions['f']['d'] / 2)) * ratio_w):int((self.dimensions[ 'f'][ 'p'] + ( self.dimensions['f']['d'] / 2)) * ratio_w) + 1, :] = 2

            df = data_nodal_variables.loc[data_nodal_variables['Simulation'] == sim]
            n_steps = len(df['Frame_increment'].unique()) - self.start_step
            n = torch.from_numpy(np.reshape(self.mesh_n, self.mesh_n.shape[0] * self.mesh_n.shape[1] * self.mesh_n.shape[2]))
            self.n_particles = n.shape[0]
            f = torch.zeros((self.n_particles, 1))
            f[n == 2] = int(sim.split('_')[6]) / 100

            # Derivate velocities from position
            total_pos = []
            for i in range(n_steps):
                step = i + self.start_step
                df_ = df.loc[df['Frame_increment'] == step]
                total_pos.append(np.asarray(df_)[:, 6:9])
            total_pos = np.asarray(total_pos)

            vel_x = []
            vel_y = []
            vel_z = []
            for particle in range(self.n_particles):
                vel_x.append(savgol_filter(np.gradient(total_pos[:, particle, 0]), 5, 3))
                vel_y.append(savgol_filter(np.gradient(total_pos[:, particle, 1]), 5, 3))
                vel_z.append(savgol_filter(np.gradient(total_pos[:, particle, 2]), 5, 3))
            vel_x = np.asarray(vel_x)
            vel_y = np.asarray(vel_y)
            vel_z = np.asarray(vel_z)

            # Complet the dataset
            for i in range(n_steps - 1):
                step = i + self.start_step
                # df_x0 = df.loc[df['Frame'] == (step * self.sampling_factor) + 1]
                df_x = df.loc[df['Frame_increment'] == step]
                df_y = df.loc[df['Frame_increment'] == step + 1]

                # y = torch.from_numpy(self.reagrupate(np.asarray(df_y), np.asarray(df_x)[:, 4:6])).to(torch.float32)
                x_ = torch.from_numpy(self.reagrupate_3D(np.asarray(df_x), vel_x[:, i], vel_y[:, i], vel_z[:, i])).to(torch.float32)
                y_ = torch.from_numpy(self.reagrupate_3D(np.asarray(df_y), vel_x[:, i + 1], vel_y[:, i + 1], vel_z[:, i + 1])).to(torch.float32)

                pos = x_[:, :3].clone()
                # pos[:, 2] = pos[:, 2] * 0
                newedge_index = compute_connectivity(np.asarray(pos), self.radius_connectivity, add_self_edges=False)
                data_total.append(Data(x_, edge_index=newedge_index, y=y_, n=n, f=f))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X'), ax.set_ylabel('Y')
            ax.set_aspect('equal')
            scatter = ax.scatter(x_[:, 0], x_[:, 2], x_[:, 1], c=n, cmap='viridis', marker='o')
            plt.savefig(sim+'.png')
        return data_total

    # import matplotlib.pyplot as plt
    # # # Crear una figura 3D
    # fig = plt.figure()
    # data = self.mesh_n
    #
    # fig.set_aspect('equal')
    #
    # x, y, z = data.shape
    # ax = fig.add_subplot(111, projection='3d')
    # x_data, y_data, z_data = np.meshgrid(range(x), range(y), range(z), indexing='ij')
    # scatter = ax.scatter(x_data, y_data, z_data, c=data.flatten(), cmap='viridis', marker='o')
    # fig.colorbar(scatter, ax=ax, label='Valores')
    # plt.show()
    def reagrupate2(self, row):
        pos = row[:, 6:9].astype(float)
        pos[:, 0] = pos[:, 0] - min(pos[:, 0])
        n = row[:, -1].astype(int)
        vel = row[:, 10:13].astype(float)
        e = row[:, 13].reshape((len(n), 1)).astype(float)
        # sigma = row[:, -7:-4]
        # tao = row[:, -4:]

        x = np.concatenate((pos, vel, e, n.reshape((len(n), 1))), axis=1)
        return x

    def calculate_glass_dataset(self, train_flag=True):
        print('Loading edges data...')
        self.start_step = 0
        self.radius_connectivity = 0.0045
        self.sampling_factor = 2

        data_total = []
        for file in self.path_to_data_folder.rglob('**/*.txt'):
            print(file)
            # Extract the simulation name from the file path
            simulation_name = file.parts[-2]
            csv_path = os.path.join(os.path.dirname(file), f'{simulation_name}.csv')
            data_nodal_variables = pd.read_csv(csv_path)

            df = pd.DataFrame(columns=list(data_nodal_variables.columns)+['n'])
            if self.desired_regions != 'all':
                for reg in self.desired_regions:
                    df_region = data_nodal_variables.loc[data_nodal_variables['Region'] == reg]
                    df_region['n'] = self.desired_regions[reg]
                    df = pd.concat([df, df_region], ignore_index=True)


            n_steps = len(df['Frame_increment'].unique()) - self.start_step -1
            n_steps = int((len(df['Frame_increment'].unique()) - self.sampling_factor) / self.sampling_factor) - self.start_step
            data_sim = []
            steps_name = df['Frame_increment'].unique()
            for i in range(n_steps):
                step = i + self.start_step
                # df_x = df.loc[df['Frame_increment'] == steps_name[step]]
                # df_y = df.loc[df['Frame_increment'] == steps_name[step + 1]]
                df_x = df.loc[df['Frame_increment'] == steps_name[(step * self.sampling_factor)] ]
                df_y = df.loc[df['Frame_increment'] == steps_name[(step * self.sampling_factor) + self.sampling_factor]]
                x = torch.from_numpy(self.reagrupate2(np.asarray(df_x))).to(torch.float32)
                y = torch.from_numpy(self.reagrupate2(np.asarray(df_y))).to(torch.float32)
                n = x[:, -1].unsqueeze(1)
                pos = x[:, :3].clone()
                newedge_index = compute_connectivity(np.asarray(pos), self.radius_connectivity, add_self_edges=False)
                data_sim.append(Data(x[:, :-1], edge_index=newedge_index, y=y[:, :-1], n=n[:, 0]))

            if train_flag:
                data_sim = random.sample(data_sim, k=int(len(data_sim) / 1))
            data_total += data_sim
        return data_total


    def save(self, path):
        # Method to save the final dataset DataFrame as a CSV file
        self.dataset_df.to_csv(path)
        print(f'Data stored as .csv at {path}')

    def write_txt_info(self, simulation_names):
        # Save dataset information in a text file
        print('Save info dataset in txt...')

        path_txt = f'data/{self.dataset_name}_info.txt'
        with open(path_txt, 'w') as archivo:
            fecha_creacion = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            archivo.write(f"Fecha de creación: {fecha_creacion}\n\n")
            archivo.write(f"Número de simulaciones: {str(len(simulation_names))} \n")
            archivo.write(f"Separación train/val: {str(self.train_split)}/{str(1-self.train_split)} \n")
            archivo.write(f"Sampling factor: {str(self.sampling_factor)} \n")
            archivo.write(f"Connectivity radio: {str(self.radius_connectivity)} \n")
            archivo.write(f"\n")
            archivo.write("\n".join(simulation_names))

    def save_dataset(self, data_total, simulation_names=[], train_flag=True):
        # Split the data into training and validation sets and save them as files
        if train_flag:
            data_total = random.sample(data_total, k=int(len(data_total) / 1))
            dataT_val = data_total[0:int(len(data_total) * (1-self.train_split))]
            dataT_train = data_total[int(len(data_total) * (1-self.train_split)):]
            torch.save(dataT_train, f'data/{self.dataset_name}_train.pt')
            torch.save(dataT_val, f'data/{self.dataset_name}_val.pt')
        else:
            torch.save(data_total, f'data/{self.dataset_name}_test.pt')

        # save info in a txt file
        self.write_txt_info(simulation_names)

