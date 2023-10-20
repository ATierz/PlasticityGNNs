import pandas as pd
import re
from pathlib import Path
import argparse


class GeometryModifier(object):
    L = 1
    R = 4

    def __init__(self, input_file_path, output_path=None, L=1., R=4):
        # Constructor to initialize object attributes
        self.L = L
        self.R = R
        self.input_file_path = Path(input_file_path)
        self.output_path = self.input_file_path.parent / 'outputs' if (len(self.input_file_path.parts) > 1
                                                                       and output_path is None) else Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

    def generate(self, L_new, R_new, output_path=None):
        # Method to generate a new geometry based on L_new and R_new
        print(f'Generating new geometry with shape L={L_new} and R={R_new}...')

        with open(self.input_file_path, 'r') as input_file:
            inside_section = False  # Initialize a flag to indicate if we're inside the relevant section
            section_lines, lines = [], []  # Create lists to store lines

            for line in input_file:  # Iterate through each line in the input file
                if '** Job' in line:
                    lines.append(f'** Job name: rectangle_L{L_new}_R{R_new} Model name: rectangle_L{L_new}_R{R_new}\n')
                    continue
                if '*Node' in line:
                    lines.append(line)
                    inside_section = True  # Start capturing lines when '*Node' is found

                elif '*Element' in line:
                    # Get new geometry, format it, and append it to lines
                    df_new_geometry = self.get_geometry(section_lines, L_new, R_new)
                    str_df_new_geometry_inp_format = GeometryModifier.set_df_to_str_inp_format(df_new_geometry)
                    lines.append(str_df_new_geometry_inp_format)
                    lines.append('\n' + line)
                    inside_section = False  # Stop capturing lines when '*Element' is found

                elif inside_section:
                    section_lines.append(line.strip().split(','))  # If inside the section, add the line to the list
                else:
                    lines.append(line)

        self.save_inp(lines, L_new, R_new)  # Save the modified content to a new INP file
        print('Done!')
        return self.output_path

    def loop_through_geometries(self, Ls, Rs):
        for L, R in zip(Ls, Rs):
            inp_paths = self.generate(L, R)
        return inp_paths

    def save_inp(self, lines, L_new, R_new):
        # Save the modified content to a new INP file
        filename_output = f'rectangle_L{L_new}_R{R_new}.inp'
        output_file_path = self.output_path / f'rectangle_L{L_new}_R{R_new}'
        output_file_path.mkdir(parents=True, exist_ok=False)
        output_file_path = output_file_path / filename_output

        with open(output_file_path, 'w') as output_file:
            output_file.writelines(lines)

        print(f'New geometry ".inp" file successfully saved at location: "{output_file_path}".')

    def get_geometry(self, section_lines, L_new, R_new):
        # Process and modify the geometry data
        df = pd.DataFrame(section_lines, columns=['index', 'x', 'y'])
        df['index'] = df['index'].astype(int)
        df['x'] = df['x'].astype(float)
        df['y'] = df['y'].astype(float)
        kx = L_new / self.L
        ky = R_new / self.R
        df['x'] = df['x'] * kx
        df['y'] = df['y'] * ky
        return df

    @staticmethod
    def set_df_to_str_inp_format(df):
        # Convert DataFrame to INP format string
        df_string = df.to_string(index=False, header=False)
        df_string_no_duplicates_spaces = re.sub(' +', ' ', df_string)
        df_string_no_duplicates_spaces_inp_format = df_string_no_duplicates_spaces.replace(' ', ',')
        df_string_no_duplicates_spaces_inp_format = df_string_no_duplicates_spaces_inp_format.replace('\n,', '\n')[1:]
        return df_string_no_duplicates_spaces_inp_format


if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Abaqus Geometry Modifier')
    parser.add_argument('--L', type=float, default=1, help='Value for L_new (default: 1.3)')
    parser.add_argument('--R', type=float, default=4, help='Value for R_new (default: 3)')
    parser.add_argument('--input_file', type=str)
    args = parser.parse_args()  # Parse command-line arguments

    geometric_modifier = GeometryModifier(args.input_file)
    geometric_modifier.generate(args.L, args.R)  # Call the generate method with parsed arguments
