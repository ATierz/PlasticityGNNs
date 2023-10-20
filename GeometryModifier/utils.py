import os


def clean_undesired_files(directory):
    # Define the allowed file extensions
    allowed_extensions = ['.odb', '.inp', '.txt', '.py']

    # Iterate through files in the directory and delete files with disallowed extensions
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_extension = os.path.splitext(filename)[1]
            if file_extension not in allowed_extensions:
                try:
                    os.remove(file_path)
                    print(f'Deleted: {file_path}')
                except Exception as e:
                    print(f'Error deleting {file_path}: {e}')