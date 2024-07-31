# Step 4: Collect the results from all ensembles and generate the final segmentation output along with uncertainty map for each image

%matplotlib inline
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv
import pandas as pd
import pydicom

# User defined threshold
threshold = 0.0001

# Define the base directory where the ensemble folders are located
base_dir = '/share/dept_machinelearning/Faculty/Rasool, Ghulam/Shared Resources/Pancreatic Cancer Image Data/result_files/nn_unet_sm_additional_unprocessed'

# Specify the directory containing the DICOM folders
dicom_directory = '/share/dept_machinelearning/Faculty/Rasool, Ghulam/Shared Resources/Pancreatic Cancer Image Data/result_files/axial_series_groups_unprocessed'

# Get a list of all directories in the specified directory
dicom_folders = [f.path for f in os.scandir(dicom_directory) if f.is_dir()]

# Get the list of files in the first ensemble folder
first_ensemble_dir = os.path.join(base_dir, 'ensemble_1_0')
file_names = [fn for fn in os.listdir(first_ensemble_dir) if fn.endswith('.npz')]

# if final_output folder does not exist
if not os.path.exists(os.path.join(base_dir, 'final_output_HU_refined')):
    os.makedirs(os.path.join(base_dir, 'final_output_HU_refined'))


# Open the CSV file
with open('/share/dept_machinelearning/Faculty/Rasool, Ghulam/Shared Resources/Pancreatic Cancer Image Data/result_files/nn_unet_sm_additional_unprocessed/final_output_HU_refined/PancreaticCancer_L3_results_add_unprocessed.csv', 'w', newline='') as csvfile:
    fieldnames = ['dicom_file_path', 'filename', 'uncertain_pixel_count', 'mean_variance', 'median_variance', 'mean_variance_percent', 
                  'median_variance_percent', 'sm_pixels', 'sm_area', 'sm_volume', 'sm_hu', 'study_description', 'series_description']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    k = 0
    # Loop over each file
    for file_name in file_names:
        k = k+1
        # Initialize a list to store the image data from each ensemble
        ensemble_data = []
        # Loop over each ensemble folder
        for j in range(1,3):
            for i in range(5):
                # Define the path to the current ensemble folder
                ensemble_dir = os.path.join(base_dir, f'ensemble_{j}_{i}')

                # Define the path to the current file
                file_path = os.path.join(ensemble_dir, file_name)

                # Check if the file is a .npz file
                if file_path.endswith('.npz'):
                    # Load the file as a numpy array
                    file_data = np.load(file_path)
                    data = file_data['probabilities']
                    background = data[0].squeeze()
                    sm_seg = data[1].squeeze()
                    # Add the data to the list
                    ensemble_data.append(sm_seg)
                           
        # Compute the average and variance of the ensemble data
        ensemble_average = np.mean(ensemble_data, axis=0)
        ensemble_variance = np.var(ensemble_data, axis=0)
        predicted_sm = np.where(ensemble_average > 0.5, 1, 0)
       
        filename = file_name.split('.')[0]
        print("file name: ", filename)

        # Get dicom series from filename
        # Replace 's', 'd', and 'groupgroup' with '_'
        filename = filename.replace('s', '_')
        filename = filename.replace('groupgroup', '_group')
        filename = filename.replace('d', '_')
        # split on last '_' and keep the part before the last '_'
        filename_parts = filename.rsplit('_', 1)
        print('Filename parts:', filename_parts)
        # Get the slice number from the filename. The slice number is the part after the last '_'
        slice = filename.rsplit('_', 1)[1]

        # Get the file_path that contains this filename from dicom_folder
        dicom_file_path = None
        for folder in dicom_folders:
            if filename_parts[0] in folder:
                dicom_file_path = folder
                break
        print('DICOM file path:', dicom_file_path)

        # Read the DICOM files and sort them by the InstanceNumber attribute
        if dicom_file_path is not None:
            files = sorted(os.listdir(dicom_file_path), key=lambda x: pydicom.dcmread(os.path.join(dicom_file_path, x)).InstanceNumber)
        else:
            print("Skipping this file due to invalid filename.")
            continue

        # Convert slice to integer
        slice_number = int(slice)

        # Get the DICOM file corresponding to the slice
        if slice_number <= len(files):
            dicomslice_file = files[slice_number]
        else:
            print("Slice number is greater than the number of DICOM files.")
            dicomslice_file = None
          
        # Check if dicomslice_file is not None before constructing the file path
        if dicomslice_file is not None:
            # Construct the full file path
            dicomslice_file_path = os.path.join(dicom_file_path, dicomslice_file)
            # Load the DICOM file
            dicom_file = pydicom.dcmread(dicomslice_file_path)
        else:
            print("Skipping this file due to invalid slice number.")
            continue  # Skip the rest of the loop for this file

        # Get the pixel dimension field data from the DICOM file header
        pixel_spacing = getattr(dicom_file, 'PixelSpacing', [0.0001, 0.0001])
        slice_thickness = getattr(dicom_file, 'SliceThickness', 0.0001)
        study_description = getattr(dicom_file, 'StudyDescription', 'No_StudyDescription')
        series_description = getattr(dicom_file, 'SeriesDescription', 'No_SeriesDescription')
        sm_area = np.sum(predicted_sm) * pixel_spacing[0] * pixel_spacing[1]
        sm_pixels = np.sum(predicted_sm)
        sm_volume = sm_area * slice_thickness
        print('Skeletal Muscle Area:', sm_area, 'mm^2')
        print('Skeletal Muscle Volume:', sm_volume, 'mm^3')
        # get the average hounsefield of the segmented area
        dicom_img_hu = get_pixels_hu(dicom_file)
        sm_hu = np.mean(dicom_img_hu[predicted_sm == 1])
        print("min sm hu value", np.min(dicom_img_hu[predicted_sm == 1]), "max sm hu value", np.max(dicom_img_hu[predicted_sm == 1]))
        print("min HU: ", np.min(dicom_img_hu), "max HU: ", np.max(dicom_img_hu))
        print('Avg Skeletal Muscle HU:', sm_hu)

        # Threshold the variance to identify uncertain pixels
        var_threshold = np.where(ensemble_variance > threshold, ensemble_variance, 0)
        non_zero_values = var_threshold[np.nonzero(var_threshold)]
        print('Count of values greater than: ', threshold, 'is', len(non_zero_values))
        print('mean variance: ', np.mean(non_zero_values), 'median variance: ', np.median(non_zero_values))

         # Normalize the thresholded variance to the range [0, 100]
        ensemble_variance_percent = ((non_zero_values - non_zero_values.min()) 
                            * (100 / (non_zero_values.max() - non_zero_values.min()))).astype(np.uint8)     
        print('mean variance percent: ', np.mean(ensemble_variance_percent), 'median variance percent: ', np.median(ensemble_variance_percent))
                
        count = len(non_zero_values)
        mean_variance = np.mean(non_zero_values)
        median_variance = np.median(non_zero_values)
        mean_variance_percent = np.mean(ensemble_variance_percent)
        median_variance_percent = np.median(ensemble_variance_percent)
        
        # Write a row to the CSV file
        writer.writerow({'dicom_file_path': dicom_file_path, 'filename': filename,'uncertain_pixel_count': count, 
                            'mean_variance': mean_variance, 
                            'median_variance': median_variance,
                            'mean_variance_percent': mean_variance_percent,
                            'median_variance_percent': median_variance_percent,
                            'sm_pixels': sm_pixels,
                            'sm_area': sm_area, 'sm_volume': sm_volume, 'sm_hu': sm_hu,
                            'study_description': study_description, 'series_description': series_description})

        predicted_sm_ = np.uint8(predicted_sm) * 255
        # Normalize the variance to the range [0, 255]
        ensemble_variance_norm = ((ensemble_variance - ensemble_variance.min()) 
                            * (255 / (ensemble_variance.max() - ensemble_variance.min()))).astype(np.uint8)  
       
        # Save the average and variance as images
        Image.fromarray(predicted_sm_).save(os.path.join(base_dir, 'final_output_HU_refined', f'prediction_{filename}.png'))
        Image.fromarray(ensemble_variance_norm).save(os.path.join(base_dir, 'final_output_HU_refined', f'uncertainty_{filename}.png'))         

        # Clear the ensemble_data list for the next ensemble
        ensemble_data.clear()    
