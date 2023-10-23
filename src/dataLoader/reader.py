import pandas as pd
import re


class DataReader(object):

    """
    DataReader class for extracting data from a specific format in text files
    """
    def __init__(self):
        # Define regular expression patterns to match lines with desired data
        self.pattern_line = r'\d+\s+[\d.E-]+\s+[\d.E-]+\s+[\d.E-]'
        self.pattern_frame_increment = r'Frame: Increment\s+(\d+)'
        self.pattern_step_time = r'Step Time =\s+([\d.E-]+)'

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

        # Convert columns to appropriate data types
        #df_tables = df_tables.astype({'Node_Label': int, 'U_Magnitude': float, 'U_U1': float, 'U_U2': float, 'Step_time': float})

        # Return the resulting DataFrame
        return df_tables
