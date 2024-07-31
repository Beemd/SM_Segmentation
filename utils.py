def generate_dicom_path_dict(axial_series_dir):
    dicom_path_dict = {}
    for root, dirs, files in os.walk(axial_series_dir):
        for dir in dirs:
            dicom_path_dict[dir] = os.path.join(root, dir)
    return dicom_path_dict

def get_dicom_path(nifti_parts, dicom_path_dict):
    return dicom_path_dict.get(nifti_parts)

def get_pixels_hu(scans):
    image = scans.pixel_array
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans.RescaleIntercept
    slope = scans.RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)
