from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import re


class DataReader(object):

    """
    DataReader class for extracting data from a specific format in text files
    """
    def __init__(self):
        # Define regular expression patterns to match lines with desired data for simulation_variable_data.txt
        self.pattern_line = r'\d+\s+[\d.E-]+\s+[\d.E-]+\s+[\d.E-]'
        self.pattern_frame_increment = r'Frame: Increment\s+(\d+)'
        self.pattern_step_time = r'Step Time =\s+([\d.E-]+)'
        # Define regular expression pattern for simulation.inp files
        self.pattern_start_element_info = r'*Element'
        self.pattern_end_element_info = r'*Nset'

    def get_edges_from_inp(self, filename):
        # Initialize empty lists to store data
        data_edges = {'element': [], 'edges': []}
        # Open the text file and read it line by line
        with open(filename, 'r') as file:

            reading_data = False  # Flag to indicate when we are reading data

            for line in file:

                if self.pattern_start_element_info in line:
                    reading_data = True
                    num_elem = 0
                    continue

                elif self.pattern_end_element_info in line:
                    break

                elif reading_data:

                    num_elem += 1

                    element_data = [int(match.group()) for match in re.finditer(r'\d+', line.strip())]

                    element_nodes = sorted(element_data[1:])
                    try:
                        elements_diagonal = ((element_nodes[0], element_nodes[-1]), (element_nodes[1], element_nodes[-2]))
                    except Exception:
                        print('hi')
                    element_fully_connected_edges = list(set(combinations(sorted(element_nodes), 2)))
                    element_edges = [item for item in element_fully_connected_edges if   # To have the index according to torch_geometric range(0,num_nodes-1)
                                              item not in elements_diagonal]
                    data_edges['element'].append(element_data[0])
                    data_edges['edges'] += element_edges

        data_edges['edge_index'] = self.build_edge_index_matrix(list(set(data_edges['edges'])))

        return data_edges

    def build_edge_index_matrix(self, edges):

        src_row, dest_row = [], []
        for elem in edges:
            src_row.append(elem[0] - 1)
            dest_row.append(elem[1] - 1)
        return [src_row, dest_row]

    def get_df_from_txt(self, filename):
        # Extract and structure data from a text file into a pandas DataFrame

        # Initialize empty lists to store data
        data = []
        column_headers = None

        # Open the text file and read it line by line
        with open(filename, 'r') as file:
            reading_data = False  # Flag to indicate when we are reading data
            current_data = []     # Temporary list to store data for one table

            for line in file:
                # Extract the Step Time from the "Frame" line
                step_time_match = re.search(self.pattern_step_time, line)
                frame_increment_match = re.search(self.pattern_frame_increment, line)
                if step_time_match:
                    current_step_time = float(step_time_match.group(1))
                if frame_increment_match:
                    current_frame_increment = int(frame_increment_match.group(1))
                # Check for lines containing column headers
                if column_headers is None and re.search(r'^\s+Node Label', line):
                    # Use the line after column headers to get the actual headers
                    column_headers = re.split(r'\s{2,}', line.strip())
                    continue

                if re.search(self.pattern_line, line):
                    # When a line matches the data pattern and we have column headers, store it in current_data list
                    current_data.append(line.strip())
                elif reading_data:
                    # When we are already reading data and an empty line is encountered,
                    # it indicates the end of the data for the current table
                    df = pd.DataFrame([data_line.split() for data_line in current_data], columns=column_headers)
                    df.insert(0, 'Frame_increment', current_frame_increment)
                    df.insert(1, 'Step_time', current_step_time)
                    data.append(df)  # Store the current table data
                    current_data = []  # Reset current_data
                    reading_data = False
                elif '---------------------------------------------------------' in line.strip():
                    # When the separator line is encountered, it indicates the start of data
                    reading_data = True

        # Create a pandas DataFrame from the extracted data
        df_tables = pd.concat(data)

        # Cast the data into floats
        for col in df_tables.columns:
            if col in ['Node Label', 'Frame_increment']:
                continue
            df_tables[col] = df_tables[col].astype(float)

        # Return the resulting DataFrame
        return df_tables

    # def get_nodes_coord_from_inp(self, filename):
    #
    #     with open(filename, 'r') as input_file:
    #         inside_section, first_node = False, True  # Initialize a flag to indicate if we're inside the relevant section
    #         section_lines = []  # Create lists to store lines
    #         for line in input_file:  # Iterate through each line in the input file
    #             if '*Node' in line and 'Output' not in line:
    #                 inside_section = True  # Start capturing lines when '*Node' is found
    #
    #             elif '*Element,' in line and 'Output' not in line:
    #                 # Get new geometry, format it, and append it to lines
    #                 df_nodes_coord = pd.DataFrame(section_lines, columns=['Node Label', 'x', 'y'])
    #
    #                 df_nodes_coord['Node Label'] = df_nodes_coord['Node Label'].astype(int)
    #                 df_nodes_coord['x'] = df_nodes_coord['x'].astype(float)
    #                 df_nodes_coord['y'] = df_nodes_coord['y'].astype(float)
    #
    #                 return df_nodes_coord
    #
    #             elif inside_section:
    #                 section_lines.append(line.strip().split(','))  # If inside the section, add the line to the list
    #
    # def compute_coord_nodes_from_displacements(self, df, nodes_coord_undeformed):
    #
    #     df_displacements = df[df['Node Label'].isin(nodes_coord_undeformed['Node Label'])]
    #     nodes_coord_undeformed['x'] = nodes_coord_undeformed['x'] + df_displacements['x']
    #     nodes_coord_undeformed['y'] = nodes_coord_undeformed['y'] + df_displacements['y']
    #
    #     df['COORD.X'] = nodes_coord_undeformed['x']
    #     df['COORD.Y'] = nodes_coord_undeformed['y']
    #
    #     return df, nodes_coord_undeformed



