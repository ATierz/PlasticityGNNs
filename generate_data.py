from src.dataGeneration.processor_geometry_inp_files import GeometryModifier
from src.dataGeneration.processor_odb_files import ODBGenerator, DataExtractorFromODB
from src.dataGeneration.utils import clean_artifact_files, store_as_txt
from src.constants import Ls, Rs

from pathlib import Path
import argparse
import time
from src import dataGeneration
import os


if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Abaqus Geometry Modifier')
    parser.add_argument('--input_file', type=str, default=r'C:\Users\mikelmartinez\Desktop\data\default.inp')
    parser.add_argument('--output_path', type=str, default=r'C:\Users\mikelmartinez\Desktop\data\outputs_final')

    args = parser.parse_args()  # Parse command-line arguments

    output_path = Path(args.output_path)

    # Store output_path as txt for ODBGenerator
    store_as_txt(str(output_path))

    # Modify Geometries and generate new
    print('\nGenerating new geometries...')
    GeometryModifier(args.input_file, output_path=output_path).loop_through_geometries(Ls, Rs)

    # Generate ODBs from INPs
    print('\nGenerating ODB files...')
    ODBGenerator.generate_odb_from_inp_files(output_path)

    # Time break to let some processes finish on Abaqus
    print('\nTime break, Abaqus is closing files...')
    time.sleep(10)

    # Extract data from OBDs, such as, U, S
    print('\nExtracting data from ODBs...')
    DataExtractorFromODB.generate_data_from_odb_files(output_path)

    # Cleaning artifact files
    print('\nCleaning artifacts...')
    for item in output_path.iterdir():
        clean_artifact_files(item)
    clean_artifact_files(os.path.dirname(dataGeneration.__file__))

