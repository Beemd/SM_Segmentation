# STEP 1: Reads dicom folders and identifies axial series and groups the files based on their attributes. 
# Created new folders for each axial series group.
# Passes the grouped dicom files to the Total Segmentator to get nifti files with all organs identified.
import os
import pydicom
import csv
import pandas as pd
import subprocess
import tempfile
import shutil
from collections import defaultdict

# Specify the directory containing the CT scans
directory = '/share/dept_machinelearning/Faculty/Rasool, Ghulam/Shared Resources/Pancreatic Cancer Image Data/10R23000239/SUBJECTS'

# Iterate over all folders in the directory
for dir in os.listdir(directory):
    # Construct the full path of the folder
    folder_path = os.path.join(directory, dir)

    # Iterate over all subfolders
    for root, dirs, files in os.walk(folder_path):
        for sub_dir in dirs:
            # Construct the full path of the subfolder
            sub_folder_path = os.path.join(root, sub_dir)

            # Count the number of DICOM files in the subfolder
            num_dicom_files = sum([1 for file in os.listdir(sub_folder_path) if file.endswith('.dcm')])

            # Only process the subfolder if it contains more than 8 DICOM files
            if num_dicom_files <= 8:
                continue

            # Group the DICOM files in the subfolder
            groups = group_dicom_files(sub_folder_path)

            for i, group in enumerate(groups.values()):
                # Extract new folder name from the complete file path
                split_path = sub_folder_path.split(os.sep)
                new_folder_name = "_".join([split_path[-5], split_path[-1], f"group{i+1}"])
              
                # Create a directory for this group in the output directory
                group_dir = os.path.join(output_directory, new_folder_name)
                os.makedirs(group_dir, exist_ok=True)

                # Copy the DICOM files for this group to the new directory
                for _, dicom_path in group:
                    shutil.copy2(dicom_path, group_dir)

                # Set the output_name based on the path to the subfolder and the group index
                split_name = sub_folder_path.split('/')
                output_name = f"nnUNet_total_seg_l3_{split_name[-5]}_{split_name[-1]}_group{i+1}"

                # Set the output_string
                output_string = '/share/dept_machinelearning/Faculty/Rasool, Ghulam/Shared Resources/Pancreatic Cancer Image Data/result_files/nifti_files_additional_unprocessed/' + output_name

                # The command to run the TotalSegmentator
                command = f"TotalSegmentator -i \"{temp_dir}\" -o \"{output_string}\" --ml"

                # Set the CUDA_VISIBLE_DEVICES environment variable
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU

                # Use subprocess to run the command and capture the output
                process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, env=env)
                out, err = process.communicate()

                # Print the output file name
                print("file_name: ", output_name)
                print(out.decode('utf-8'))
