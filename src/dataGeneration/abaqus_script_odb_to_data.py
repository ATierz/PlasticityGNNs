'''
    This script appears to open an Abaqus output database (ODB), iterate over frames, and write the nodal
    displacements (U) to a text file for each frame. It is assumed that the script is run within the directory
    containing the ODB file of interest. If you want to extract additional data or make other modifications,
    you can adapt the script as needed.
'''

# TODO: For better generalization change the way of naming the output path. Now i can not provide parameter path, so it's
# TODO: fixed in line 43 to where it should look. Default outputs

from abaqus import *
from abaqusConstants import *
from viewerModules import *
from driverUtils import executeOnCaeStartup
import argparse


#################SOME FUNCTIONS#########################
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
##############################################################


# Create a viewport and set it as the current one
session.Viewport(name='Viewport: 1',
             origin=(0.0, 0.0),
             width=253.28515625,
             height=127.874992370605)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
executeOnCaeStartup()

# Call the function to start looping through folders
odb_files = find_odb_files(r'C:\Users\mikelmartinez\Desktop\data\outputs')

for odb_file in odb_files:
    # Open the identified .odb file
    odb_session = session.openOdb(name=odb_file)

    # Set the current displayed object in the viewport to the opened .odb
    session.viewports['Viewport: 1'].setValues(displayedObject=odb_session)

    # Access the .odb file using its name
    odb = session.odbs[odb_file]

    # Extracting Step 1, assuming there is only one step
    step1 = odb_session.steps.values()[0]


    # Loop over all frames in Step 1
    for count, i in enumerate(odb_session.steps[step1.name].frames):

        # Set the frame to display in the viewport
        session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=count)

        # Configure field report options
        session.fieldReportOptions.setValues(printTotal=OFF,
                                             printMinMax=OFF,
                                             numberFormat=nf)

        # Write a field report to a text file
        session.writeFieldReport(
            fileName=os.path.splitext(odb_file)[0] + '_variables_data.txt',
            append=ON,
            sortItem='Element Label',
            odb=odb,
            step=0,
            frame=count,
            outputPosition=NODAL,
            variable=(
                ('U', NODAL),  # Nodal displacements
                ('S', INTEGRATION_POINT),  # Nodal tension
                ('COORD', NODAL)
            ))

    # Close the .odb file
    odb.close()


