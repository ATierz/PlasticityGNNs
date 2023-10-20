from geometry_modifier import GeometryModifier
from odb_generator import ODBGenerator, DataExtractorFromODB
from utils import clean_undesired_files
import argparse
import time

Rs = [1, 1.25, 1.5, 1.75, 2]
Ls = [4, 3.5, 3, 2.75, 2]

if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Abaqus Geometry Modifier')
    parser.add_argument('--input_file', type=str)
    args = parser.parse_args()  # Parse command-line arguments
    # Modify Geometries and generate new
    print('\nGenerating new geometries...')
    output_path = GeometryModifier(args.input_file).loop_through_geometries(Ls, Rs)
    # Generate ODBs from INPs
    print('\nGenerating ODB files...')
    ODBGenerator.generate_odb_from_inp_files(output_path)
    # Time break to let some processes finish on Abaqus
    time.sleep(15)
    # Extract data from OBDs, such as, U, S
    print('\nExtracting data from ODBs...')
    DataExtractorFromODB.generate_data_from_odb_files(output_path.parent, variables=['U', 'S'])

    clean_undesired_files(output_path)
