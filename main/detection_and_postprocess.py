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

    closest_contour_mask = np.zeros_like(mask)
    if closest_contour is not None:
        cv2.drawContours(closest_contour_mask, [closest_contour], -1, (1), thickness=cv2.FILLED)

    return closest_contour_mask

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

        mask = np.zeros((height, width), dtype=np.uint8)

        with open(annotation_path, 'r') as file:
            for line in file:
                parts = line.split()
                coords = list(map(float, parts[1:-1]))
                
                if len(coords) % 2 != 0:
                    raise ValueError("Coordinate pairs are incomplete.")
                
                points = np.array([(int(x * width), int(y * height))
                                for x, y in zip(coords[0::2], coords[1::2])], dtype=np.int32)
                
                if len(points) != 0:
                    cv2.fillPoly(mask, [points], color=255)
        return mask

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise

    except Exception as e:
        print(f"An error occurred while generating detection mask: {e}")
        raise

def resize_mask(mask, height, width):
    """
    Resizes the mask to match the dimensions of the flux map.
    """
    # flux_map_height, flux_map_width = flux_map.shape
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_CUBIC)

def get_combined_mask(building_mask, detection_mask, display=False):
    if building_mask.shape != detection_mask.shape:
        raise ValueError("Masks must have the same dimensions.")
    # Combine the masks of same shape
    combined_mask = np.logical_and(building_mask, detection_mask).astype(np.uint8)

    if display:
        # Visualize the results using subplots
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # Ensures axs is a 2x2 array of axes

        # Visualization of the building mask
        axs[0, 0].imshow(building_mask, cmap='gray')
        axs[0, 0].set_title('Building Mask')
        axs[0, 0].axis('on')

        axs[0, 1].imshow(detection_mask, cmap='gray')
        axs[0, 1].set_title('Detection Mask')
        axs[0, 1].axis('on')

        # Visualization of the initial combined mask before resizing
        axs[1, 0].imshow(combined_mask, cmap='gray')
        axs[1, 0].set_title('Combined Mask')
        axs[1, 0].axis('on')

        # Visualization of detection results
        axs[1, 1].axis('off')

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()
    
    return combined_mask

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

def get_masked_monthly_flux(combined_mask, flux_map):
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
    # Apply the combined resized mask to the flux map
    try:
        masked_flux_map = flux_map * combined_mask
        summed_monthly_flux = np.sum(masked_flux_map)
        return summed_monthly_flux
    except combined_mask.shape != flux_map.shape:
        raise ValueError("Mask and flux map must have the same dimensions.")
    except Exception as e:
        print(f"An error occurred while computing masked monthyly flux: {e}")
        raise

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

def process_address(address_path, detection_results_path, results_df, detected_addresses, single_building=False):
    """
    Process detection results and corresponding solar information for a given address 
    and updates the results DataFrame.

    Args:
    address_path (str): Path to the directory of the address being processed, containing geotiff files.
    results_df (pd.DataFrame): DataFrame where results are stored.
    """
    address_name = os.path.basename(address_path)
    if address_name in detected_addresses:
        img_name = f'rgb_{address_name}'
        ann_path = os.path.join(detection_results_path, f'{img_name}.txt')
        img_path = os.path.join(detection_results_path, '..', '..', 'images', f'{img_name}.jpg')
        # Load necessary data
        with rasterio.open(os.path.join(address_path, f'monthlyFlux_{address_name}.tif')) as src_flux, \
             rasterio.open(os.path.join(address_path, f'mask_{address_name}.tif')) as src_mask, \
             rasterio.open(os.path.join(address_path, f'rgb_{address_name}.tif')) as src_img:
            
            building_mask = src_mask.read(1)  # Assuming mask is single-band
            if single_building:
                main_building_mask = get_main_building_mask(building_mask)
            else:
                main_building_mask = building_mask
            
            detection_mask = generate_detection_mask(src_img.name, ann_path)

            # Use the first month's flux map to determine the dimensions for resizing
            sample_flux_map = src_flux.read(1)
            
            # Resize main_building_mask and detection_mask if they don't match the flux map size
            if main_building_mask.shape != sample_flux_map.shape:
                main_building_mask = resize_mask(main_building_mask, sample_flux_map.shape[0], sample_flux_map.shape[1])
            if detection_mask.shape != sample_flux_map.shape:
                detection_mask = resize_mask(detection_mask, sample_flux_map.shape[0], sample_flux_map.shape[1])
            
            combined_mask = get_combined_mask(main_building_mask, detection_mask, display=False)
            visualize_results(img_path, combined_mask)
            panel_area = count_area(combined_mask)

            monthly_flux = []
            for month in range(1, 13):
                month_flux_map = src_flux.read(month)
                
                # Ensure combined_mask matches the current month_flux_map dimensions
                if combined_mask.shape != month_flux_map.shape:
                    combined_mask = resize_mask(combined_mask, month_flux_map.shape[0], month_flux_map.shape[1])
                
                flux = get_masked_monthly_flux(combined_mask, month_flux_map)
                monthly_flux.append(flux)
            
            annual_flux = sum(monthly_flux)
            results_df.loc[len(results_df)] = [address_name, panel_area] + monthly_flux + [annual_flux]

def visualize_results(img_path, combined_mask):
    """
    Visualizes the combined mask over the image, draws a bounding box around the masked area, 
    and saves the result.

    Args:
    img_path (str): Path to the original image file.
    combined_mask (numpy.ndarray): The combined mask to overlay on the image.
    """
    img_name = os.path.basename(img_path)
    save_img_dir = os.path.join(os.path.dirname(img_path), '..', 'detection_results', 'images_masked')
    os.makedirs(save_img_dir, exist_ok=True)
    save_img_path = os.path.join(save_img_dir, f"masked_{img_name}")

    # Read the image
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"The image at {img_path} could not be loaded.")
    
    # Ensure the mask is the same size as the image
    if combined_mask.shape != image.shape[:2]:
        combined_mask = resize_mask(combined_mask, image.shape[0], image.shape[1])
    
    # Normalize the mask to ensure it is 0 or 1
    combined_mask = combined_mask / np.max(combined_mask)
    
    # Convert the combined mask to a color overlay (red)
    mask_color = np.zeros_like(image)
    mask_color[:, :, 2] = combined_mask * 255  # red channel

    # Create a semi-transparent overlay by blending the mask with the image
    overlay = cv2.addWeighted(image, 1.0, mask_color, 0.3, 0)

    # Find contours of the combined mask to draw the bounding box
    contours, _ = cv2.findContours(np.uint8(combined_mask * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding box around the contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Red bounding box

    # Save the result
    cv2.imwrite(save_img_path, overlay)

def run_detection_and_analysis(data_root, model_path, save_img=False, save_crop=False, batch_size=30, conf=0.1, iou=0.7, img_size=640, process_grid=False):
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
        process_address(address_path, detection_results_path, results_df, detected_addresses, single_building=process_grid)

    # Save the DataFrame
    results_df.to_csv(os.path.join(data_root, 'solar_flux_results.csv'), index=False)
    print("Results saved to solar_flux_results.csv")