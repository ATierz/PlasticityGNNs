from pathlib import Path
import pandas as pd
from src.dataLoader.reader import DataReader


class DataLoader(object):

    """
    Class with a load method for loading data from ".txt" files within subfolders of a specified data folder.
    It allows you to specify desired simulations and reads the data using the DataReader class. The results are
    stored in a pandas DataFrame, and a 'Simulation' column is added to indicate the simulation name.
    Finally, the individual DataFrames are concatenated into a single dataset DataFrame.
    """
    def __init__(self, path_to_data_folder):
        # Initialize the DataLoader with the path to the data folder
        self.path_to_data_folder = Path(path_to_data_folder)
        self.dataset_df = None

    def load(self, desired_simulations=None):
        # Load data from text files in the data folder

        print('Loading data...')

        # Initialize an empty list to store DataFrames for each simulation
        dataset_df = []

        # Loop through all ".txt" files within subfolders of the data folder
        for file in self.path_to_data_folder.rglob('**/*.txt'):

            # Extract the simulation name from the file path
            simulation_name = file.parts[-2]

            # Check if the simulation is in the list of desired simulations
            if desired_simulations is not None:
                if simulation_name not in desired_simulations:
                    continue  # Skip this simulation if not desired

            # Use the DataReader to read the data from the text file into a DataFrame
            simulation_df = DataReader().get_df_from_txt(file)

            # Insert a new column 'Simulation' with the simulation name
            simulation_df.insert(0, 'Simulation', simulation_name)

            # Append the simulation DataFrame to the dataset list
            dataset_df.append(simulation_df)

        # Concatenate the list of DataFrames into a single DataFrame
        self.dataset_df = pd.concat(dataset_df) if len(dataset_df) > 0 else print('Empty DataFrame, no data found to concatenate.')

        print('Done!')

        return self.dataset_df

    def save(self, path):
        self.dataset_df.to_csv(path)
        print(f'Data stored as .csv at {path}')
