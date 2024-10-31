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

def save_dicom(predicted_mask, dicom_file, output_path):
    # Create a new DICOM file based on the original DICOM file
    new_dicom = dicom_file.copy()

    # Ensure the pixel data is in the correct format and scale it properly
    predicted_mask = (predicted_mask * 255).astype(np.uint8)  # Scale to 0-255 for visibility

    # Update the pixel data with the predicted mask
    new_dicom.PixelData = predicted_mask.tobytes()

    # Update necessary metadata fields to match the predicted mask
    new_dicom.Rows, new_dicom.Columns = predicted_mask.shape
    new_dicom.BitsAllocated = 8  # Set BitsAllocated to match uint8 data
    new_dicom.SamplesPerPixel = 1  # Grayscale image
    new_dicom.PhotometricInterpretation = "MONOCHROME2"

    # Update SOPInstanceUID to make it unique
    new_dicom.SOPInstanceUID = pydicom.uid.generate_uid()
    new_dicom.SeriesInstanceUID = pydicom.uid.generate_uid()
    new_dicom.InstanceNumber = 1
    new_dicom.SeriesDescription = "Predicted Mask"
    new_dicom.ImageType = ["DERIVED", "PRIMARY"]
    new_dicom.ContentDate = datetime.datetime.now().strftime('%Y%m%d')
    new_dicom.ContentTime = datetime.datetime.now().strftime('%H%M%S')

    # Set window center and window width to ensure the mask is visible
    new_dicom.WindowCenter = 128
    new_dicom.WindowWidth = 256

    # Ensure the correct transfer syntax
    new_dicom.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # Save the new DICOM file
    new_dicom.save_as(output_path)
