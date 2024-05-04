#!/usr/bin/env python
# coding: utf-8

# Initialize the necessary libraries and variables
import os
import pandas as pd
from geopy.geocoders import GoogleV3
import requests
from IPython.display import Image, display
from PIL import Image as im
from pathlib import Path
import io
import numpy as np
import json
import subprocess
from osgeo import gdal
import xarray as xr
import rasterio
from PIL import Image
import math

### Change these accordingly:
SOURCE_FOLDER = 'D:\Capstone Data\Capstone Dataset Spreadsheets' # input folder containing spreadsheets to parse
DEST_PATH = 'D:\Capstone Data' # destination for output folders
RADIUS = 70 # data radius in meters. google static maps api output images are about 280 meters across (diameter)
COORDS = False # Whether you are using inputted coordinates to form a grid or address lists to query specific places
FAILED_LOCATIONS = "Failed Google Solar Locations" # name for failed locations file. Do not add file type at the end
ALL_IMAGES = "Output Images" # Same here but for the folder containing all of the output images in one place
PIXEL_SIZE_METERS = 0.25 # Number of pixel corresponding to a meter in the image. Default 0.25

origin_coord = (47.62051365556474, -122.34927846329165) # (Latitude, longitude) format
height = 3 # number of boxes tall the grid is
width = 3 # number of boxes wide the grid is

# Google API key
api_key = 
geolocator = GoogleV3(api_key)

# Alters failed addresses document title to be unique and not overwrite another file
failed_locations = FAILED_LOCATIONS.replace(" ", "_")
i = 0
while os.path.exists(os.path.join(DEST_PATH, f"{failed_locations}_{i}.txt")):
    i = i + 1
failed_locations_path = os.path.join(DEST_PATH, f"{failed_locations}_{i}.txt")

all_images = ALL_IMAGES.replace(" ", "_")
i = 0
while os.path.exists(os.path.join(DEST_PATH, f"{all_images}_{i}")):
    i = i + 1
all_images_path = os.path.join(DEST_PATH, f"{all_images}_{i}")
os.makedirs(all_images_path)


# Calculates the meter to degree longitude conversion scalar using the latitude
# Required because Earth is not a perfect sphere, so difference in longitude relies on latitude
def meter_to_long(latitude):
    return 1 / ((math.pi / 180) * 6378137 * math.cos(latitude * math.pi / 180))

# latitude (N/S) then longitude (E/W)
def generate_grid(origin_coord, height, width):
    # Fix non integer paramters
    height = math.ceil(height)
    width = math.ceil(width)
    meter_to_lat = 1 / ((math.pi / 180) * 6378000)
    # Convert origin coord to top left box's coord
    starting_lat = float(origin_coord[0]) + (height/2 - 0.5) * meter_to_lat * RADIUS * 2
    starting_long = float(origin_coord[1]) - (width/2 - 0.5) * meter_to_long(starting_lat) * RADIUS * 2
    grid_coords = []
    for i in range (0, height):
        for j in range (0, width):
            lat = starting_lat - i * meter_to_lat * RADIUS * 2
            long = starting_long + j * meter_to_long(lat) * RADIUS * 2
            grid_coords.append((lat, long))
    return grid_coords
            

def convert_geotiff_to_png(geotiff_path, jpg_path):
    geotiff_file = rasterio.open(geotiff_path)
    band1 = np.array(geotiff_file.read(1))
    band2 = np.array(geotiff_file.read(2))
    band3 = np.array(geotiff_file.read(3))
    numpy_image = np.dstack((band1, band2, band3))
    im = Image.fromarray(numpy_image)
    for path in jpg_path:
        im.save(path)

def request_and_save(request_url, solar_folder, location_name, latitude, longitude):
    print("Request URL: " + request_url)
    print("Location: " + location_name)
    response = requests.get(request_url)
    if response.status_code == 200:
        if not os.path.exists(solar_folder):
            os.makedirs(solar_folder)
        # saves response

        new_data = {
            "image_coord": f"{latitude} {longitude}",
            "imageRes": f"{PIXEL_SIZE_METERS}",
            "imageAreaCoverage": f"{RADIUS}",
            "dsm_fname": f"dsm_{location_name}.tif",
            "mask_fname": f"mask_{location_name}.tif",
            "monthlyFlux_fname": f"monthlyFlux_{location_name}.tif"
        }

        json_path = os.path.join(solar_folder, f"ResponseJSON_{location_name}.json")
        with open(json_path, "wb") as json_file:
            json_file.write(response.content)

        with open(json_path, 'r') as json_file:
            existing_data = json.load(json_file)
        existing_data.update(new_data)

        with open(json_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

        # tags = ["dsmUrl", "rgbUrl", "maskUrl", "annualFluxUrl", "monthlyFluxUrl"]
        tags = ["dsmUrl", "rgbUrl", "maskUrl", "monthlyFluxUrl"]
        data = json.loads(response.content)
        for tag in tags:
            if tag in data:
                url = data[tag]
                geotiff_path = os.path.join(solar_folder, f"{tag[:-3]}_{location_name}.tif")
                response = requests.get(url + f"&key={api_key}")
                if response.status_code == 200:
                    with open(geotiff_path, "wb") as geotiff_file:
                        geotiff_file.write(response.content)
                else:
                    print(f"Failed to download GeoTiff {tag} for address: {location_name}")

                if tag == "rgbUrl":
                    image_name = f"jpg_{location_name}.jpg"
                    jpg_path = [os.path.join(solar_folder, image_name), os.path.join(all_images_path, image_name)]
                    convert_geotiff_to_png(geotiff_path, jpg_path)
            else:
                print(f"Failed to retrieve the {tag} tag for location: {location_name}")
    else:
        print(f"Failed to retrieve the Google Solar API response for location: {location_name}")
        # Prints the failed addresses to a text file in the dest_folder path
        with open(failed_locations_path, "a") as f:
            f.write(location_name + "\n")
            f.close()


def processSpreadsheet(file):
    df = pd.read_csv(os.path.join(SOURCE_FOLDER, file))
    addresses = df['ADDRESS']
    cities = df['CITY']
    states = df['STATE OR PROVINCE']
    zipCodes = df['ZIP OR POSTAL CODE']

    if len(addresses) != len(cities) or len(cities) != len(states) or len(states) != len(zipCodes):
        print(file + "has empty address features and will be skipped.")
        return

    csv_folder_name = file[:-4]
    if not os.path.exists(os.path.join(DEST_PATH, csv_folder_name)):
        os.makedirs(os.path.join(DEST_PATH, csv_folder_name))

    for address, city, state, zipCode in zip(addresses, cities, states, zipCodes):
        if zipCode > 0:
            zipCode = int(zipCode)
        full_address = f"{address} {city} {state} {zipCode}"
        location = geolocator.geocode(full_address)
        if location:
            latitude = location.latitude
            longitude = location.longitude

            request_url_data = f"https://solar.googleapis.com/v1/dataLayers:get?location.latitude={latitude}&location.longitude={longitude}&radiusMeters={RADIUS}&view=FULL_LAYERS&requiredQuality=HIGH&pixelSizeMeters={PIXEL_SIZE_METERS}&key={api_key}"
            address_for_filename = full_address.replace('/', '-') # gets rid of / so it wont be confused as a file path
            address_for_filename = full_address.replace(' ', '_') # changes delimiter to _
            solar_folder = os.path.join(DEST_PATH, csv_folder_name, address_for_filename)
            request_and_save(request_url_data, solar_folder, address_for_filename, latitude, longitude)
        else:
            print(f"Coordinates not found for the address: {full_address}")
        print()

def processCoordinates(coord_list, folder_name):
    if not os.path.exists(os.path.join(DEST_PATH, folder_name)):
        os.makedirs(os.path.join(DEST_PATH, folder_name))
    for coord in coord_list:
        latitude = coord[0]
        longitude = coord[1]
        request_url_data = f"https://solar.googleapis.com/v1/dataLayers:get?location.latitude={latitude}&location.longitude={longitude}&radiusMeters={RADIUS}&view=FULL_LAYERS&requiredQuality=HIGH&pixelSizeMeters={PIXEL_SIZE_METERS}&key={api_key}"
        solar_folder = os.path.join(DEST_PATH, folder_name, f"{latitude}_{longitude}")
        request_and_save(request_url_data, solar_folder, f"{latitude}_{longitude}", latitude, longitude)

if COORDS:
    coord_list = generate_grid(origin_coord, height, width)
    processCoordinates(coord_list, f"Grid_H={height}_W={width}_{origin_coord[0]}_{origin_coord[1]}")
else:
    for filename in os.listdir(SOURCE_FOLDER):
        processSpreadsheet(filename)




