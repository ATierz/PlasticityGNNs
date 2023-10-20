import subprocess
from pathlib import Path
import os


class ODBGenerator(object):

    @staticmethod
    def find_inp_files(folder):
        odb_files = []  # List to store found .odb files
        for item in folder.iterdir():
            if item.is_dir():
                # If it's a subfolder, dive into it and extend the list with the files found in it
                subfolder_odb_files = ODBGenerator.find_inp_files(item)
                odb_files.extend(subfolder_odb_files)
            elif item.is_file() and item.suffix == '.inp':
                print("Found .inp file:", item)
                odb_files.append(item)  # Add the file to the list
        return odb_files

    @staticmethod
    def generate_odb_from_inp_files(path):
        paths_inp_files = ODBGenerator.find_inp_files(Path(path))

        for path_inp in paths_inp_files:
            os.chdir(path_inp.parent)

            cmd = f"abaqus job={path_inp.stem} input={path_inp.parts[-1]}"
            subprocess.run(cmd, shell=True)


class DataExtractorFromODB(object):

    @staticmethod
    def find_odb_files(folder):
        odb_files = []  # List to store found .odb files
        for item in folder.iterdir():
            if item.is_dir():
                # If it's a subfolder, dive into it and extend the list with the files found in it
                subfolder_odb_files = DataExtractorFromODB.find_odb_files(item)
                odb_files.extend(subfolder_odb_files)
            elif item.is_file() and item.suffix == '.odb':
                print("Found .odb file:", item)
                odb_files.append(item)  # Add the file to the list
        return odb_files

    @staticmethod
    def generate_data_from_odb_files(path):
        os.chdir(path)
        cmd = f"abaqus cae noGui=generate_data_from_odb.py"
        subprocess.run(cmd, shell=True)









