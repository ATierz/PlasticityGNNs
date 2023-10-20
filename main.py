from src.processor_geometry_inp_files import GeometryModifier
from src.processor_odb_files import ODBGenerator, DataExtractorFromODB
from src.utils import clean_artifact_files

import argparse
import time
import src
import os

Rs = [1, 1.25, 1.5, 1.75, 2, 2.5]
Ls = [4, 3.5, 3, 2.75, 2, 1.75]

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
    time.sleep(10)
    # Extract data from OBDs, such as, U, S
    print('\nExtracting data from ODBs...')
    DataExtractorFromODB.generate_data_from_odb_files(output_path)
    # Cleaning artifact files
    print('\nCleaning artifacts...')
    for item in output_path.iterdir():
        clean_artifact_files(item)
    clean_artifact_files(os.path.dirname(src.__file__))
