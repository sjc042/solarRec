"""
This script performs detection and post-processing of detection results 
along with relevant solar information.
"""
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import argparse
import pandas as pd

import torch
from ultralytics import YOLO


def get_main_building_mask(mask):
    """
    Finds the contour closest to the center of the image from a binary mask and returns a mask with only that contour.

    Args:
    mask (np.ndarray): A binary mask array where the building areas are expected to be 1s.

    Returns:
    np.ndarray: A mask with the closest contour to the center of the image filled, value 0/1.
    """
    # Convert the binary mask to a format suitable for findContours
    mask_bin = np.uint8(mask * 255)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate the center of the image
    image_center = np.array([mask.shape[1] / 2, mask.shape[0] / 2])
    
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

    # Create a blank mask to draw the closest contour
    closest_contour_mask = np.zeros_like(mask)

    # If a closest contour was found, draw it on the mask
    if closest_contour is not None:
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

    Returns:
    np.ndarray: A mask where polygons are filled based on the annotations.

    Raises:
    FileNotFoundError: If the image or annotation file cannot be found.
    ValueError: If there are incomplete coordinate pairs in the annotations.
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
                # Expected line format: class_id x1 y1 x2 y2 ... xn yn conf
                parts = line.split()
                
                # Extract coordinates, ignoring class_id at the start and conf at the end
                coords = list(map(float, parts[1:-1]))
                
                # Check if coordinates form complete pairs
                if len(coords) % 2 != 0:
                    raise ValueError("Coordinate pairs are incomplete in the annotations file.")
                
                # Convert normalized coordinates to actual image coordinates
                points = np.array([(int(x * width), int(y * height))
                                   for x, y in zip(coords[0::2], coords[1::2])], dtype=np.int32)

                # Draw the polygon on the mask
                cv2.fillPoly(mask, [points], color=255)
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
    mask (numpy.ndarray): Numpy array containing values 0 and 1 or 0 and 255, where 1 or 255 indicates the presence of the area to be measured.
    res (float): Resolution of the mask in meters per pixel.

    Returns:
    float: The total area in square meters represented by the True (1 or 255) pixels in the mask.

    Raises:
    ValueError: If the mask contains values other than [0, 1] or [0, 255].
    """
    unique_values = np.unique(mask)
    
    # Check if the mask is in the correct form or only contains a single value (0 or 255)
    if set(unique_values).issubset({0, 1}):
        # Mask already in binary form with 0 and/or 1
        pixel_count = np.sum(mask)  # Sum all 1's
    elif set(unique_values).issubset({0, 255}):
        # Mask values in form of 0 and/or 255
        pixel_count = np.sum(mask == 255)
    else:
        # If the mask contains any other values, raise an error
        raise ValueError("Mask values must be either [0, 1] or [0, 255]. Found values: " + str(unique_values))

    # Calculate the total area
    area = pixel_count * (res ** 2)
    return area

def get_masked_monthly_flux(mask, flux_map):
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
    # building_mask_height, building_mask_width = building_mask.shape
    # detection_mask_height, detection_mask_width = detection_mask.shape
    # flux_map_height, flux_map_width = flux_map.shape

    # # Resize masks if their dimensions do not match the flux map
    # if (building_mask_height != flux_map_height or building_mask_width != flux_map_width):
        
    #     # FIXME: use get_main_building_mask for generating building mask
    #     building_mask = cv2.resize(building_mask, (flux_map_width, flux_map_height), interpolation=cv2.INTER_NEAREST)
    # if (detection_mask_height != flux_map_height or detection_mask_width != flux_map_width):
        
    #     # FIXME: detection mask is not correctly generated
    #     detection_mask = cv2.resize(detection_mask, (flux_map_width, flux_map_height), interpolation=cv2.INTER_NEAREST)

    # # Combine the resized masks
    # combined_mask_resized = np.logical_and(building_mask, detection_mask).astype(np.uint8)

    # Apply the combined resized mask to the flux map
    masked_flux_map = flux_map * mask

    # if display:
    #     # Visualize the results using subplots
    #     fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # Ensures axs is a 2x2 array of axes

    #     # Visualization of the building mask
    #     axs[0, 0].imshow(building_mask, cmap='gray')
    #     axs[0, 0].set_title('Building Mask after resizing')
    #     axs[0, 0].axis('on')

    #     # Visualization of the masked monthly flux map
    #     image_masked_flux_map = axs[0, 1].imshow(masked_flux_map, cmap='viridis')
    #     fig.colorbar(image_masked_flux_map, ax=axs[0, 1], label='Flux (kWh/kW/year)')
    #     axs[0, 1].set_title('Combined Masked Monthly Flux Map')
    #     axs[0, 1].axis('on')

    #     # Visualization of the initial combined mask before resizing
    #     axs[1, 0].imshow(combined_mask_resized, cmap='gray')
    #     axs[1, 0].set_title('Combined Mask after resizing')
    #     axs[1, 0].axis('on')

    #     # Visualization of detection results
    #     axs[1, 1].imshow(detection_mask, cmap='gray')
    #     axs[1, 1].set_title('Detection mask')
    #     axs[1, 1].axis('on')

    #     plt.tight_layout()  # Adjust layout to prevent overlap
    #     plt.show()

    summed_monthlyFlux = np.sum(masked_flux_map)

    return summed_monthlyFlux

def get_combined_mask(building_mask, detection_mask, flux_map, display=False):
    # Get dimensions of input masks and flux map
    building_mask_height, building_mask_width = building_mask.shape
    detection_mask_height, detection_mask_width = detection_mask.shape
    flux_map_height, flux_map_width = flux_map.shape

    # Resize masks if their dimensions do not match the flux map
    if (building_mask_height != flux_map_height or building_mask_width != flux_map_width):
        
        # FIXME: use get_main_building_mask for generating building mask
        building_mask = cv2.resize(building_mask, (flux_map_width, flux_map_height), interpolation=cv2.INTER_NEAREST)
    if (detection_mask_height != flux_map_height or detection_mask_width != flux_map_width):
        
        # FIXME: detection mask is not correctly generated
        detection_mask = cv2.resize(detection_mask, (flux_map_width, flux_map_height), interpolation=cv2.INTER_NEAREST)

    # Combine the resized masks
    combined_mask_resized = np.logical_and(building_mask, detection_mask).astype(np.uint8)

    if display:
        # Visualize the results using subplots
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # Ensures axs is a 2x2 array of axes

        # Visualization of the building mask
        axs[0, 0].imshow(building_mask, cmap='gray')
        axs[0, 0].set_title('Building Mask after resizing')
        axs[0, 0].axis('on')

        # Visualization of the masked monthly flux map
        masked_flux_map = flux_map * combined_mask_resized
        image_masked_flux_map = axs[0, 1].imshow(masked_flux_map, cmap='viridis')
        fig.colorbar(image_masked_flux_map, ax=axs[0, 1], label='Flux (kWh/kW/year)')
        axs[0, 1].set_title('Combined Masked Monthly Flux Map')
        axs[0, 1].axis('on')

        # Visualization of the initial combined mask before resizing
        axs[1, 0].imshow(combined_mask_resized, cmap='gray')
        axs[1, 0].set_title('Combined Mask after resizing')
        axs[1, 0].axis('on')

        # Visualization of detection results
        axs[1, 1].imshow(detection_mask, cmap='gray')
        axs[1, 1].set_title('Detection mask')
        axs[1, 1].axis('on')

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()
    
    return count_area(combined_mask_resized), combined_mask_resized

def estimate_monthly_flux(address_path, ann_path, month, display_mask=False):
    """
    Calculates the monthly solar flux for a specified address based on detection results and visualizes the 
    results if required.

    Args:
    address_path (str): Path to the directory of the address being processed, containing geotiff files.
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
    address = os.path.basename(address_path)
    # Construct file paths
    file_names = {
        "monthly_flux": f"monthlyFlux_{address}.tif",
        "building_mask": f"mask_{address}.tif",
        "rgb_image": f"rgb_{address}.tif"           # TODO: need to change image path
    }
    files = {key: os.path.join(address_path, value) for key, value in file_names.items()}

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
            main_building_mask = get_main_building_mask(building_mask)

        detection_mask = generate_detection_mask(files['rgb_image'], ann_path)
        panel_area, combined_mask = get_combined_mask(main_building_mask, detection_mask, month_flux_map, display=display_mask)
        # panel_area = count_area(detection_mask)
        
        masked_monthly_flux = get_masked_monthly_flux(combined_mask, month_flux_map)
        masked_monthly_flux = masked_monthly_flux.sum()
    
    except rasterio.errors.RasterioIOError as e:
        raise FileNotFoundError(f"Rasterio failed to open files: {str(e)}")
    except Exception as e:
        raise Exception(f"An error occurred during processing: {str(e)}")

    return panel_area, masked_monthly_flux

def detect_solar_panel(model_path, img_dir,
                        save_dir=None, save_img=False, 
                        save_crop=False, batch_size=30, 
                        conf=0.1, iou=0.7, img_size=640):
    """
    Run YOLO model prediction on a specified image directory.

    Args:
    model_path (str): Path to the YOLO model checkpoint file.
    img_dir (str): Image directory for prediction.
    save_img (bool): Whether to save images with predictions.
    save_crop (bool): Whether to save detected instances cropped from images.
    batch_size (int): Batch size for processing images.
    conf (float): Confidence threshold for detection.
    iou (float): IOU threshold for detection.
    img_size (int): Image size for processing.

    Returns:
    None
    """
    # Determine the computing device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple silicon mps device.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using Nvidia CUDA device.")
    else:
        device = "cpu"
        print("Using CPU.")

    # Check if the model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

    # Check if the image directory exists
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory '{img_dir}' not found.")

    # model_exp_name = 'yolo' + model_path.split('yolo')[1].split('/')[0].split('.pt')[0]
    # split = os.path.basename(os.path.dirname(img_dir))
    # exp_name = '_'.join(["predict", split, f'conf-{conf}', f'iou-{iou}', model_exp_name])
    exp_name = 'detection_results'

    source = os.path.join(img_dir, '*.jpg')

    if not save_dir:
        # save_dir = os.path.join(img_dir, "..", "detection_results")
        save_dir = os.path.join(img_dir, "..")
        os.makedirs(save_dir, exist_ok=True)

    # Load pretrained model
    model = YOLO(model_path)

    # Use the model to predict
    model.predict(source=source, name=exp_name, device=device, 
                            batch=batch_size, imgsz=img_size,
                            iou=iou, conf=conf, save_txt=True, 
                            save_conf=True, save=save_img, save_crop=save_crop,
                            project=save_dir, show_labels=False)
    results_dir = os.path.join(save_dir, exp_name, 'labels')
    return results_dir

def process_address(address_path, detection_results_path, results_df, detected_addresses):
    """
    Process detection results along with solar information for a given address 
    and updates the results DataFrame.

    Args:
    address_path (str): Path to the directory of the address being processed, containing geotiff files.
    results_df (pd.DataFrame): DataFrame where results are stored.
    """
    address_name = os.path.basename(address_path)
    if address_name in detected_addresses:
        img_name = f'rgb_{address_name}'
        ann_path = os.path.join(detection_results_path, f'{img_name}.txt')
        # img_path = os.path.join(address_path, '..', '..', 'images', f'{img_name}.jpg')
        # detection_mask = generate_detection_mask(img_path, ann_path)  # mask values: [0, 255]
        # FIXME: need to get panel area in combined mask of detection and building
        monthly_flux = []
        for month in range (1, 13):
            # TODO: after debugging mask, set display_mask to False
            panel_area, flux = estimate_monthly_flux(address_path, ann_path, month, display_mask=True)
            monthly_flux.append(flux)
        annual_flux = sum(monthly_flux)
        results_df.loc[len(results_df)] = [address_name] + [panel_area] + monthly_flux + [annual_flux]

# TODO: add individual address masked detetction visualization
def visualize_results(ann_path, img_path, mask_path):
    with rasterio.open(mask_path) as src_mask:
        building_mask = src_mask.read(1)  # Assuming mask is single-band


def run_detection_and_analysis(data_root, model_path, save_img=False, save_crop=False, batch_size=30, conf=0.1, iou=0.7, img_size=640):
    """
    Run YOLO model prediction and process detection results with relevant solar information.

    Args:
    data_root (str): Root directory containing data files and image directories.
    model_path (str): Path to the YOLO model checkpoint file.
    save_img (bool): If True, saves images with predictions.
    save_crop (bool): If True, saves detected instances cropped from images.
    batch_size (int): Batch size for processing images.
    conf (float): Confidence threshold for detection.
    iou (float): IOU threshold for detection.
    img_size (int): Image size for processing.

    Raises:
    FileNotFoundError: If the specified directories do not exist.
    """
    img_dir = os.path.join(data_root, "images")
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root directory '{data_root}' not found.")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory '{img_dir}' not found.")

    # Detect solar panels and process results
    detection_results_path = detect_solar_panel(model_path, img_dir,
                                                save_dir=None, save_img=save_img, 
                                                save_crop=save_crop, batch_size=batch_size, 
                                                conf=conf, iou=iou, img_size=img_size)

    # Initialize results pandas DataFrame with columns for address, panel area, twelve months, and annual flux
    results_df = pd.DataFrame(columns=['Address', 'Panel_Area(m^2)', 'January_Flux', 'February_Flux', 'March_Flux', 'April_Flux',
                                       'May_Flux', 'June_Flux', 'July_Flux', 'August_Flux', 'September_Flux', 
                                       'October_Flux', 'November_Flux', 'December_Flux', 'Annual_Flux(kWh/kW/year)'])

    address_dir = os.path.join(data_root, 'addresses')
    detected_addresses = set(os.path.basename(address).replace('.txt', '').replace('rgb_', '') for address in os.listdir(detection_results_path))
    # Process each address
    for address in os.listdir(address_dir):
        address_path = os.path.join(address_dir, address)
        process_address(address_path, detection_results_path, results_df, detected_addresses)

    # Save the DataFrame
    results_df.to_csv(os.path.join(data_root, 'solar_flux_results.csv'), index=False)
    print("Results saved to solar_flux_results.csv")

