from src.dataGeneration.processor_geometry_inp_files import GeometryModifier
from src.dataGeneration.processor_odb_files import ODBGenerator, DataExtractorFromODB
from src.dataGeneration.utils import clean_artifact_files
from src.constants import Ls, Rs

import argparse
import time
import src
import os


if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Abaqus Geometry Modifier')
    parser.add_argument('--input_file', type=str, default=r'C:\Users\mikelmartinez\Desktop\data\necking.inp')
    parser.add_argument('--output_path', type=str, default=r'C:\Users\mikelmartinez\Desktop\data\outputs\trials')

    args = parser.parse_args()  # Parse command-line arguments

    # Modify Geometries and generate new
    print('\nGenerating new geometries...')
    output_path = GeometryModifier(args.input_file, output_path=args.output_path).loop_through_geometries(Ls, Rs)

    # Generate ODBs from INPs
    print('\nGenerating ODB files...')
    ODBGenerator.generate_odb_from_inp_files(output_path)

    # Time break to let some processes finish on Abaqus
    print('Time break, Abaqus is closing files...')
    time.sleep(10)

    # Extract data from OBDs, such as, U, S
    print('\nExtracting data from ODBs...')
    DataExtractorFromODB.generate_data_from_odb_files(output_path)

    # Cleaning artifact files
    print('\nCleaning artifacts...')
    for item in output_path.iterdir():
        clean_artifact_files(item)
    clean_artifact_files(os.path.dirname(src.__file__))

