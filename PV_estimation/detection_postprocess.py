"""
This script performs post-processing for the detection results along with relevant solar information.
"""
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def get_main_building_mask(mask_path):
    """
    Finds the contour closest to the center of the image from a binary mask and returns a mask with only that contour.

    Args:
    mask_path (str): Path to the binary mask GeoTIFF file.

    Returns:
    np.ndarray: A mask with the closest contour to the center of the image filled, value 0/1.
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        mask_bin = np.uint8(mask * 255)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_center = np.array([src.width / 2, src.height / 2])
    min_dist = np.inf
    closest_contour = None

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = np.array([cx, cy])
            dist = np.linalg.norm(centroid - image_center)
            if dist < min_dist:
                min_dist = dist
                closest_contour = contour

    # Create a blank mask and draw the closest contour filled
    closest_contour_mask = np.zeros_like(mask)
    cv2.drawContours(closest_contour_mask, [closest_contour], -1, (1), thickness=cv2.FILLED)
    
    return closest_contour_mask

def get_monthly_flux(filepath, month):
    """
    Retrieves the flux map for a specific month from a 12-band GeoTIFF file.

    Args:
    filepath (str): Path to the GeoTIFF file.
    month (int): Month to retrieve (1 for January, 2 for February, ..., 12 for December).

    Returns:
    np.ndarray: A 2D array representing the flux map of the specified month.

    Raises:
    ValueError: If the month is not within the range 1 to 12, or if the file does not contain exactly 12 bands.
    FileNotFoundError: If the GeoTIFF file cannot be accessed.
    """
    if month < 1 or month > 12:
        raise ValueError("Invalid month. Please choose a value between 1 and 12.")
    try:
        with rasterio.open(filepath) as src:
            if src.count != 12:
                raise ValueError("The GeoTIFF file does not contain 12 bands, as required for monthly data.")

            # Read the band corresponding to the specified month
            data = src.read(month)  # Month 1 corresponds to band 1, and so on
            
            return data

    except rasterio.errors.RasterioIOError as e:
        raise FileNotFoundError(f"Unable to locate or read the file at {filepath}") from e

def generate_detection_mask(image_path, annotation_path):
    """
    Generates and optionally saves a mask from polygon annotations with normalized coordinates.
    Displays the mask using matplotlib.

    Args:
    image_path (str): Path to the image for which the mask is to be created.
    annotation_path (str): Path to the text file containing normalized polygon coordinates.
    Raises:
    FileNotFoundError: If the image or annotation file cannot be found.
    Exception: For any other errors during the execution.
    """
    try:
        # Load the image to find out the dimensions using rasterio
        with rasterio.open(image_path) as src:
            width, height = src.width, src.height
        
        # Prepare a blank mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Read the annotations and draw each polygon on the mask
        with open(annotation_path, 'r') as file:
            for line in file:
                # line format: class_id x1 y1 x2 y2 ...
                coords = list(map(float, line.split()))[1:]
                if len(coords) % 2 != 0:
                    raise ValueError("Coordinate pairs are incomplete in the annotations file.")
                
                points = np.array([(int(coords[i] * width), int(coords[i + 1] * height)) 
                                   for i in range(0, len(coords), 2)], dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)  # 255 to fill the mask in white
        return mask
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"An error occurred while generating detection mask: {e}")
        raise

def count_area(mask, res=0.25):
    """
    Calculates the area represented by True (1) pixels in a binary mask, based on the resolution of the mask.

    Args:
    mask (ndarray): Numpy array containing values 0 and 1, where 1 or True indicates the presence of the area to be measured.
    res (float): Resolution of the mask in meters per pixel.

    Returns:
    float: The total area in square meters represented by the True (1) pixels in the mask.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)  # Ensure the mask is boolean for summing True values

    pix_area = np.sum(mask)  # Count of True/1 pixels in the mask

    area = pix_area * res ** 2
    return area

def get_masked_monthly_flux(building_mask, detection_mask, flux_map, display=False):
    """
    Applies a combined building and detection mask to a monthly flux map and visualizes the result.

    Args:
    building_mask (numpy.ndarray): Building mask array.
    detection_mask (numpy.ndarray): Detection mask array.
    flux_map (numpy.ndarray): Monthly flux map array.
    display (bool): if to visualize masks and results

    Returns:
    numpy.ndarray: The masked monthly flux map.
    """
    # Get dimensions of input masks and flux map
    building_mask_height, building_mask_width = building_mask.shape
    detection_mask_height, detection_mask_width = detection_mask.shape
    flux_map_height, flux_map_width = flux_map.shape

    # Resize masks if their dimensions do not match the flux map
    if (building_mask_height != flux_map_height or building_mask_width != flux_map_width):
        building_mask = cv2.resize(building_mask, (flux_map_width, flux_map_height), interpolation=cv2.INTER_NEAREST)
    if (detection_mask_height != flux_map_height or detection_mask_width != flux_map_width):
        detection_mask = cv2.resize(detection_mask, (flux_map_width, flux_map_height), interpolation=cv2.INTER_NEAREST)

    # Combine the resized masks
    combined_mask_resized = np.logical_and(building_mask, detection_mask).astype(np.uint8)

    # Apply the combined resized mask to the flux map
    masked_flux_map = flux_map * combined_mask_resized

    if display:
        # Visualize the results using subplots
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # Ensures axs is a 2x2 array of axes

        # Visualization of the combined mask after resizing
        image_combined_mask_resized = axs[0, 0].imshow(combined_mask_resized, cmap='gray')
        axs[0, 0].set_title('Combined Mask after resizing')
        axs[0, 0].axis('on')

        # Visualization of the masked monthly flux map
        image_masked_flux_map = axs[0, 1].imshow(masked_flux_map, cmap='viridis')
        fig.colorbar(image_masked_flux_map, ax=axs[0, 1], label='Flux (kWh/kW/year)')
        axs[0, 1].set_title('Masked Monthly Flux Map')
        axs[0, 1].axis('on')

        # Visualization of the initial combined mask before resizing
        axs[1, 0].axis('off')

        # Leave one subplot empty (bottom-right)
        axs[1, 1].axis('off')

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    summed_monthlyFlux = np.sum(masked_flux_map)

    return summed_monthlyFlux

def estimate_monthly_flux(root, address, ann_path, month, display_mask=False):
    """
    Calculates the monthly solar flux for a specified address based on detection results and visualizes the 
    results if required.

    Args:
    root (str): Base directory for data files.
    address (str): Address identifier, formatted as '1099_Skyevale_NE,_Ada,_MI_49301'.
    ann_path (str): Path to the file containing detection results.
    month (int): Month number (1-12) for which flux is to be calculated.
    display_mask (bool): Flag to determine whether to display masks and results.

    Returns:
    tuple: A tuple containing:
        - Total panel area (float): The total area covered by solar panels in square meters.
        - Sum of the masked monthly flux (float): The total monthly solar flux for the detected panels.
    
    Raises:
    FileNotFoundError: If any required files are not found.
    ValueError: If the month provided is not within the valid range.
    """
    if not (1 <= month <= 12):
        raise ValueError("Month must be between 1 and 12.")
    
    # Construct file paths
    file_names = {
        "monthly_flux": f"monthlyFlux_{address}.tif",
        "building_mask": f"mask_{address}.tif",
        "rgb_image": f"rgb_{address}.tif"           # TODO: need to change image path
    }
    files = {key: os.path.join(root, address, value) for key, value in file_names.items()}

    # Check for file existence
    missing_files = [name for name, path in files.items() if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")
    try:
        # Load necessary data
        with rasterio.open(files['monthly_flux']) as src_flux, \
             rasterio.open(files['building_mask']) as src_mask:
            month_flux_map = src_flux.read(month)
            building_mask = src_mask.read(1)  # Assuming mask is single-band

        detection_mask = generate_detection_mask(files['rgb_image'], ann_path)
        panel_area = count_area(detection_mask)
        
        masked_monthly_flux = get_masked_monthly_flux(building_mask, detection_mask, month_flux_map, display=display_mask)
        masked_monthly_flux_sum = masked_monthly_flux.sum()
    except rasterio.errors.RasterioIOError as e:
        raise FileNotFoundError(f"Rasterio failed to open files: {str(e)}")
    except Exception as e:
        raise Exception(f"An error occurred during processing: {str(e)}")

    return panel_area, masked_monthly_flux_sum




def test1():
    data_root = '/home/psc/Desktop/solarRec/API_Data_Generation/TestAddresses'
    address = '1099_Skyevale_NE_Ada_MI_49301'
    ann_path = '/home/psc/Desktop/PSC/Custom_Dataset/custom-dataset-2000-seg/valid/labels/1-Toll-Ln_18_quad_1_png.rf.41bc7cf400d0177429efd261cfb65b2c.txt'
    month = 7
    solarPanel_area, solarPanel_monthlyFlux = estimate_monthly_flux(data_root,
                        address,
                        ann_path,
                        month,
                        display_mask=True)

    print(f"Solar Panel Area: {solarPanel_area:.2f} m^2")
    print(f"Solar Panel Monthly flux in month {month}: {solarPanel_monthlyFlux:.2f} kWh/kW/year")

test1()

