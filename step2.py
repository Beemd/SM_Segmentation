# STEP 2: Identify L3 slices in the segmented niftii files created in step2 and converts to png.
# PNG files are saved in a new specified folder
import pydicom
from PIL import Image
import pandas as pd
import nibabel as nib
import numpy as np
import os
from utils import *

png_folder = '/share/dept_machinelearning/Faculty/Rasool, Ghulam/Shared Resources/Pancreatic Cancer Image Data/result_files/png_slices_L3_additional_unprocessed'
os.makedirs(png_folder, exist_ok=True)

axial_series_dir = '/share/dept_machinelearning/Faculty/Rasool, Ghulam/Shared Resources/Pancreatic Cancer Image Data/result_files/axial_series_groups_unprocessed'

output_folder = '/share/dept_machinelearning/Faculty/Rasool, Ghulam/Shared Resources/Pancreatic Cancer Image Data/result_files/nifti_files_additional_unprocessed/'
output_files = [f for f in os.listdir(output_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]
vertebra_value = 29 # index for L3 vertebra

# Generate the dicom_path_dict
dicom_path_dict = generate_dicom_path_dict(axial_series_dir)

# Load the nifti file
for nifti_file in output_files:
    print("nifti file: ", nifti_file)
    try:
        nifti = nib.load(os.path.join(output_folder, nifti_file))
        nifti_data = nifti.get_fdata()
        first_slice_shape = nifti_data[:,:,0].shape
        for slice_index in range(1, nifti_data.shape[2]):
            assert nifti_data[:,:,slice_index].shape == first_slice_shape, f"Invalid shape: {nifti_data[:,:,slice_index].shape} for slice {slice_index}. Expected {first_slice_shape}."
    except AssertionError as e:
        print(e)
        continue
    nifti_transposed_data = np.transpose(nifti_data, (2,0,1))
    print("nifti data shape: ", nifti_data.shape)
    for slice_ in range(nifti_data.shape[2]):
        # check if the slice contains this vertebra value:
        if vertebra_value in nifti_transposed_data[slice_, :, :]:
        # get the corresponding DICOM folder
            nifti_parts = '_'.join(nifti_file.split('_')[4:8]).replace('.nii', '')
            dicom_folder = get_dicom_path(nifti_parts, dicom_path_dict)
            if dicom_folder is None:
                print(f"No matching DICOM folder for NIfTI file {nifti_file}")
                continue
            # get a sorted list of DICOM files in the folder based on their Instance Number or Slice Location
            dicom_files = sorted(os.listdir(dicom_folder), key=lambda x: pydicom.dcmread(os.path.join(dicom_folder, x)).InstanceNumber)
            # get the corresponding DICOM file. The slices in the nifti_data are in reverse order compared to dicom files in the dicom folder
            slice_num = (nifti_data.shape[2]) - 1 - slice_
            dicom_file = dicom_files[slice_num]
            dicom_data = pydicom.dcmread(os.path.join(dicom_folder, dicom_file))
            # convert the DICOM slice to png format
            dicom_image = get_pixels_hu(dicom_data) # convert the DICOM slice to Hounsfield units
            muscle_window = np.array(dicom_image)
            muscle_window[muscle_window < -29] = -29
            muscle_window[muscle_window > 150] = 150
            image = np.uint8(muscle_window)
            image_data = Image.fromarray(image)
            # derive the filename from the nifti_file name
            base_name = nifti_file.split('_')[4] + 'd' + nifti_file.split('_')[5] + 's' + nifti_file.split('_')[6] + nifti_file.split('_')[7]
            base_name = base_name.replace('.nii', '')
            png_filename = f"{base_name}_{slice_num:04d}_0000.png"
            png_path = os.path.join(png_folder, png_filename)
            # save the DICOM slice as a PNG file
            image_data.save(png_path)
