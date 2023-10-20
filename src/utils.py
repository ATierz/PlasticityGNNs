import os


def clean_artifact_files(directory):
    # Define the allowed file extensions
    allowed_extensions = ['.odb', '.inp', '.txt', '.py', '.cae']

    # Iterate through files in the directory and delete files with disallowed extensions
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_extension = os.path.splitext(filename)[1]
            if file_extension not in allowed_extensions:
                try:
                    os.remove(file_path)
                    #print(f'Deleted: {file_path}')
                except Exception as e:
                    print(f'Error deleting {file_path}: {e}')


def find_odb_files(folder):
    odb_files = []  # List to store found .odb files
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            # If it's a subfolder, dive into it and extend the list with the files found in it
            subfolder_odb_files = find_odb_files(item_path)
            odb_files.extend(subfolder_odb_files)
        elif os.path.isfile(item_path) and item.endswith('.odb'):
            print("Found .odb file:", item_path)
            odb_files.append(item_path)  # Add the file to the list
    return odb_files
