# STEP 1: Use this code
# Reads dicom folders and identifies axial series and groups the files based on their attributes.
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

def read_dicom_attributes(dicom_path):
    """Read necessary DICOM attributes from a single DICOM file and determine orientation."""
    if not dicom_path.endswith('.dcm'):
        return None

    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    except pydicom.errors.InvalidDicomError:
        # This file is not a valid DICOM file
        return None
    
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)

    # Initialize default values for attributes that might be missing
    default_orientation = ("unknown",) * 6  # Default for ImageOrientationPatient
    default_position = (0.0, 0.0, 0.0)  # Default for ImagePositionPatient

    orientation = "unknown"  # Default value
    if hasattr(ds, 'ImageOrientationPatient') and ds.ImageOrientationPatient is not None:
        orientation_tuple = ds.ImageOrientationPatient
        row_vector = orientation_tuple[:3]
        col_vector = orientation_tuple[3:]
        cross_product = [
            row_vector[1] * col_vector[2] - row_vector[2] * col_vector[1],
            row_vector[2] * col_vector[0] - row_vector[0] * col_vector[2],
            row_vector[0] * col_vector[1] - row_vector[1] * col_vector[0],
        ]
        max_abs_value = max(cross_product, key=abs)
        if abs(max_abs_value) < 0.8:
            orientation = "oblique"
        max_index = cross_product.index(max_abs_value)
        if abs(max_abs_value) >= 0.8 and max_index == 0:
            orientation = "sagittal"
        elif abs(max_abs_value) >= 0.8 and max_index == 1:
            orientation = "coronal"
        elif abs(max_abs_value) >= 0.8 and max_index == 2:
            orientation = "axial"

    attributes = {
        "StudyInstanceUID": getattr(ds, "StudyInstanceUID", "UnknownStudy"),
        "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", "UnknownSeries"),
        "ImageOrientationPatient": orientation,
        "SliceThickness": getattr(ds, "SliceThickness", 0.0001),  # Assuming a sensible default can be 0.0001
        "SpacingBetweenSlices": getattr(ds, "SpacingBetweenSlices", None),  # This is optional and might not be present
        "ImagePositionPatient": getattr(ds, "ImagePositionPatient", default_position),
        "FrameOfReferenceUID": getattr(ds, "FrameOfReferenceUID", "UnknownFrameOfReference"),
        "Series Number": getattr(ds, "SeriesNumber", "UnknownSeriesNumber"),
        "Acquisition Number": getattr(ds, "AcquisitionNumber", "UnknownAcquisitionNumber"),
    }
    return attributes

def group_dicom_files(directory):
    """Group DICOM files based on specific attributes, considering only axial orientation."""
    groups = defaultdict(list)
    for root, _, files in os.walk(directory):
        for file in files:
            dicom_path = os.path.join(root, file)
            attrs = read_dicom_attributes(dicom_path)
            if attrs is None or attrs["ImageOrientationPatient"] != "axial":
                continue
            if attrs["ImageOrientationPatient"] == "axial":
                group_key = (root, attrs["StudyInstanceUID"], attrs["SeriesInstanceUID"],
                             attrs["SliceThickness"], attrs["SpacingBetweenSlices"],
                             attrs["FrameOfReferenceUID"], attrs["Series Number"], attrs["Acquisition Number"])
                groups[group_key].append((attrs["ImagePositionPatient"][2], dicom_path))  # Sort by Z position

    for group in groups.values():
        group.sort()

    return groups

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
