from src.dataGeneration.processor_geometry_inp_files import GeometryModifier
from src.dataGeneration.processor_odb_files import ODBGenerator, DataExtractorFromODB
from src.dataGeneration.utils import clean_artifact_files, store_as_txt
from src.constants import Ls, Rs

from pathlib import Path
import argparse
import time
from src import dataGeneration
import os
import shutil


if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Abaqus Geometry Modifier')
    parser.add_argument('--input_file', type=str, default=r'C:\Users\AMB\Documents\PhD\data\2D\Foam_2D\Foam_800_200\inp')
    parser.add_argument('--output_path', type=str, default=r'C:\Users\AMB\Documents\PhD\code\PlasticityGNNs\outputs')
    parser.add_argument('--extract_odb_flag', type=bool, default=True)

    args = parser.parse_args()  # Parse command-line arguments

    output_path = Path(args.output_path)
    if args.extract_odb_flag:
        for i in os.listdir(output_path):
            name_folder = os.path.splitext(i)[0]
            os.mkdir(os.path.join(output_path, name_folder))
            shutil.move(os.path.join(output_path, i), os.path.join(output_path, name_folder, i))
        # Modify Geometries and generate new
        # print('\nGenerating new geometries...')
        # output_path = GeometryModifier(args.input_file, output_path=args.output_path).loop_through_geometries(Ls, Rs)

        # Store output_path as txt for ODBGenerator
        store_as_txt(str(output_path))

        # Generate ODBs from INPs
        print('\nGenerating ODB files...')
        ODBGenerator.generate_odb_from_inp_files(output_path)

        # Time break to let some processes finish on Abaqus
        print('Time break, Abaqus is closing files...')
        time.sleep(15)

    # Extract data from OBDs, such as, U, S
    print('\nExtracting data from ODBs...')
    DataExtractorFromODB.generate_data_from_odb_files(output_path)

    # for i in os.listdir(output_path):
    #     folder_path = os.path.join(output_path, i)
    #     shutil.copyfile(os.path.join(folder_path, f'{i}_variables_data.txt'), os.path.join(output_path, f'{i}.txt'))
    #     shutil.rmtree(folder_path)


    print('\nCleaning artifacts...')
    for item in output_path.iterdir():
        clean_artifact_files(item)
    clean_artifact_files(os.path.dirname(dataGeneration.__file__))

