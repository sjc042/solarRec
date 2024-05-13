# Libraries for Google Solar API query
import os
import pandas as pd
from geopy.geocoders import GoogleV3
import requests
import numpy as np
from PIL import Image
import json
import rasterio
from dotenv import load_dotenv

def convert_geotiff_to_image(geotiff_path, jpg_path):
    """Convert a GeoTIFF file to a JPEG image."""
    with rasterio.open(geotiff_path) as geotiff_file:
        band1, band2, band3 = geotiff_file.read([1, 2, 3])
    # Stack the arrays along the third axis to form an RGB image
    numpy_image = np.dstack((band1, band2, band3))
    # Convert the NumPy array to a PIL Image and save it
    im = Image.fromarray(numpy_image)
    im.save(jpg_path)

def get_unique_filename(directory, base_name, extension):
    """Create a unique filename in the given directory by appending an index."""
    i = 0
    unique_path = os.path.join(directory, f"{base_name}_{i}{extension}")
    while os.path.exists(unique_path):
        i += 1
        unique_path = os.path.join(directory, f"{base_name}_{i}{extension}")
    return unique_path

# main funciton of API data query
def query_from_API(src, save_dir=None, RADIUS=70, PIXEL_SIZE_METERS=0.25):
    """Query the Google Solar API based on input source."""
    assert save_dir is not None, "Please specify save directory for list of addresses."
    
    # Load API key from environment
    load_dotenv()
    api_key = os.getenv('GOOGLE_SOLAR_API_KEY')
    if not api_key:
        raise ValueError("API Key not found. Please check your .env file.")
    print("API Key found.")

    
    failed_addresses = []  # List to track failed addresses

    def request_and_save(location_name, latitude, longitude, save_dir=None):
        """Request data from the Google Solar API and save the results."""
        print("Processing Location:", location_name)
        image_qualities = ["HIGH", "MEDIUM", "LOW"]
        for quality in image_qualities:
            request_url = f"https://solar.googleapis.com/v1/dataLayers:get?location.latitude={latitude}&location.longitude={longitude}&radiusMeters={RADIUS}&view=FULL_LAYERS&requiredQuality={quality}&pixelSizeMeters={PIXEL_SIZE_METERS}&key={api_key}"
            response = requests.get(request_url)
            if response.status_code == 200:
                # Process and save the data from a successful response
                address_dir = os.path.join(save_dir, 'addresses', location_name)
                image_dir = os.path.join(save_dir, 'images')
                os.makedirs(address_dir, exist_ok=True)
                os.makedirs(image_dir, exist_ok=True)

                # Save the response content in a JSON file
                json_path = os.path.join(address_dir, f"responseJSON_{location_name}.json")
                with open(json_path, "wb") as json_file:
                    json_file.write(response.content)

                # Deserialize JSON content to process it
                data = json.loads(response.content)
                existing_data = {
                    "imageCoord": f"{latitude} {longitude}",
                    "imageRes": str(PIXEL_SIZE_METERS),
                    "imageAreaCoverage": str(RADIUS),
                    "dsm_fname": f"dsm_{location_name}.tif",
                    "mask_fname": f"mask_{location_name}.tif",
                    "monthlyFlux_fname": f"monthlyFlux_{location_name}.tif"
                }
                existing_data.update(data)
                
                # Save the merged data back to the JSON file
                with open(json_path, 'w') as json_file:
                    json.dump(existing_data, json_file, indent=4)

                # Download and save the associated GeoTIFF files
                tags = ["dsmUrl", "rgbUrl", "maskUrl", "monthlyFluxUrl"]
                for tag in tags:
                    if tag in data:
                        url = data[tag] + f"&key={api_key}"
                        response = requests.get(url)
                        if response.status_code == 200:
                            geotiff_path = os.path.join(address_dir, f"{tag.replace('Url', '')}_{location_name}.tif")
                            with open(geotiff_path, "wb") as geotiff_file:
                                geotiff_file.write(response.content)
                            if tag == "rgbUrl":
                                image_path = os.path.join(image_dir, f"{tag.replace('Url', '')}_{location_name}.jpg")
                                convert_geotiff_to_image(geotiff_path, image_path)
                        else:
                            print(f"Failed to download GeoTiff {tag} for address: {location_name}")
                return  # Exit after successful processing

        # If all qualities fail, add address to failed list
        print(f"Failed to retrieve the Google Solar API response for all quality levels for location: {location_name}")
        failed_addresses.append(location_name)

    def parseAddressSheet(file_path):
        df = pd.read_csv(file_path, header=0)               # Assuming first row is column header
        location_list = []
        for index, row in df.iterrows():
            try:
                # Extract and check if address, city, state, and zipCode are NaN
                address, city, state, zipCode = row.get('ADDRESS'), row.get('CITY'), row.get('STATE OR PROVINCE'), row.get('ZIP OR POSTAL CODE')

                latitude = row['LATITUDE'] if 'LATITUDE' in row and pd.notna(row['LATITUDE']) else None
                longitude = row['LONGITUDE'] if 'LONGITUDE' in row and pd.notna(row['LONGITUDE']) else None

                # Use address if available; otherwise, rely on latitude and longitude
                if pd.notna(address) and pd.notna(city) and pd.notna(state) and pd.notna(zipCode):
                    spaced_address = ' '.join([address, city, state, str(int(zipCode))]).replace('/', ' ')
                    location_list.append(spaced_address)
                elif pd.notna(latitude) and pd.notna(longitude):
                    location_list.append((latitude, longitude))
                else:
                    print(f"Insufficient data at index {index} to proceed with geocoding.")
            except Exception as e:
                print(f"Error processing row index {index}: {e}")
            
        return location_list

    def parseAddressTextFile(txt_path):
        addresses = []
        
        with open(txt_path, 'r') as file:
            for line in file:
                clean_line = line.strip()
                if clean_line:  # Check if the line is not empty
                    # Try to parse the line as coordinates
                    if ',' in clean_line and all(part.strip().replace('.', '', 1).replace('-', '', 1).isdigit() for part in clean_line.split(',')):
                        # Split the line by comma and convert to a tuple of floats
                        parts = clean_line.split(',')
                        coordinates = (float(parts[0].strip()), float(parts[1].strip()))
                        addresses.append(coordinates)
                    else:
                        # Otherwise, treat it as an address string
                        addresses.append(clean_line)
        
        return addresses

    def processList(location_list, save_dir):
        geolocator = GoogleV3(api_key)
        for loc in location_list:
            if isinstance(loc, tuple):
                # process loc as coordinate (lat, long)
                latitude, longitude = loc
                location_name = f"lat{latitude:.6f}_long{longitude:.6f}"
                request_and_save(location_name, latitude, longitude, save_dir)
            elif isinstance(loc, str):
                # process loc as an address string, address in format: ' '.join(filter(None, [address, city, state, str(int(zipCode))]))
                # format the address string
                loc = loc.replace('_', ' ').replace(',', ' ').replace('/', ' ').replace('-', ' ')
                coord = geolocator.geocode(loc)
                if coord:
                    latitude = coord.latitude
                    longitude = coord.longitude
                    location_name = loc.replace('/', '-').replace('  ', '_').replace(' ', '_')
                    request_and_save(location_name, latitude, longitude, save_dir)
                else:
                    print(f"Failed to geocode address: {loc}")
            else:
                print(f"Skipping invalid location input: {loc}.")
    
    failed_fname = "Failed_Google_Solar_Locations"          # name for failed locations file. Do not add file type at the end
    # Alters failed addresses document title to be unique and not overwrite another file
    failed_fpath = get_unique_filename(save_dir, failed_fname, '.txt')
    
    if isinstance(src, list):        # input is a list of coordinates or addresses
        processList(src, save_dir=save_dir)
    elif isinstance(src, str) and src.endswith('.csv'):
        loc_list = parseAddressSheet(src)
        processList(loc_list, save_dir)
    elif isinstance(src, str) and src.endswith('.txt'):
        loc_list = parseAddressTextFile(src)
        processList(loc_list, save_dir)
    elif isinstance(src, str) or isinstance(src, tuple):       # input is a single address
        processList([src], save_dir=save_dir)
    else:
        print(f"Input source is invalid of type {type(src)}.")
        print("Allowed value types: str (csv file path or address), list (of addresses or coordinates), tuple(coordinate latitude and longitude)")

    # Write failed addresses to a file
    if failed_addresses:
        with open(failed_fpath, 'w') as f:
            for address in failed_addresses:
                f.write(f"{address}\n")
        print(f"Failed addresses have been written to {failed_fpath}")