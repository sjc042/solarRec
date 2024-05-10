'''
This script combines the three main components for the solar recognition system.
The first component is Google Solar API query based on address
second component is solar panel detection on aerial images,
third component is post-processing of detection resutls 
and estimating address-grained solar energy production.
'''

# Libraries for Google Solar API query
import os
import pandas as pd
from geopy.geocoders import GoogleV3
import requests
import numpy as np
import json
import rasterio
from PIL import Image
from dotenv import load_dotenv


'''
1. Google Solar API Query
'''
# helper functions
def convert_geotiff_to_image(geotiff_path, jpg_path):
    geotiff_file = rasterio.open(geotiff_path)
    band1 = np.array(geotiff_file.read(1))
    band2 = np.array(geotiff_file.read(2))
    band3 = np.array(geotiff_file.read(3))
    numpy_image = np.dstack((band1, band2, band3))
    im = Image.fromarray(numpy_image)
    im.save(jpg_path)

def get_unique_filename(directory, base_name, extension):
    i = 0
    unique_path = os.path.join(directory, f"{base_name}_{i}{extension}")
    while os.path.exists(unique_path):
        i += 1
        unique_path = os.path.join(directory, f"{base_name}_{i}{extension}")
    return unique_path

# main funciton of API data query
def query_from_API(src, data_dir=None, api_key=None, RADIUS=70, PIXEL_SIZE_METERS=0.25):
    # TODO: 
    # 1. check source type
    # 2. assert api_key is not None
    # 3. assert pixel_size_meters in (0.1, 0.25, 0.5, 1.0) values in float

    def request_and_save(location_name, latitude, longitude, save_dir):
        request_url = f"https://solar.googleapis.com/v1/dataLayers:get?location.latitude={latitude}&location.longitude={longitude}&radiusMeters={RADIUS}&view=FULL_LAYERS&requiredQuality=HIGH&pixelSizeMeters={PIXEL_SIZE_METERS}&key={api_key}"
        # print("Request URL: " + request_url)
        print("Processing Location: " + location_name)
        response = requests.get(request_url)
        if response.status_code == 200:             # success
            address_dir = os.path.join(save_dir, 'addresses', location_name)       # assume location delimeted by '_'
            image_dir = os.path.join(save_dir, 'images')
            # TODO: optionally check and make dirs earlier on before this
            if not os.path.exists(address_dir):
                os.makedirs(address_dir)
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            
            # saves response
            new_data = {
                # NOTE: may save location/address name?
                "imageCoord": f"{latitude} {longitude}",
                "imageRes": f"{PIXEL_SIZE_METERS}",
                "imageAreaCoverage": f"{RADIUS}",
                "dsm_fname": f"dsm_{location_name}.tif",
                "mask_fname": f"mask_{location_name}.tif",      # building mask
                "monthlyFlux_fname": f"monthlyFlux_{location_name}.tif"
            }

            json_path = os.path.join(address_dir, f"responseJSON_{location_name}.json")
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
                    response = requests.get(url + f"&key={api_key}")
                    if response.status_code == 200:         # success
                        geotiff_path = os.path.join(address_dir, f"{tag.replace('Url', '')}_{location_name}.tif")
                        with open(geotiff_path, "wb") as geotiff_file:
                            geotiff_file.write(response.content)
                        if tag == "rgbUrl":
                            image_path = os.path.join(image_dir, f"{tag.replace('Url', '')}_{location_name}.jpg")
                            convert_geotiff_to_image(geotiff_path, image_path)
                    else:
                        print(f"Failed to download GeoTiff {tag} for address: {location_name}")
                else:
                    print(f"Failed to retrieve the {tag} tag for location: {location_name}")
        else:
            print(f"Failed to retrieve the Google Solar API response for location: {location_name}")
            # Prints the failed addresses to a text file in the dest_folder path
            with open(failed_fpath, "w") as f:
                f.write(location_name + "\n")
                f.close()
    
    def processSpreadsheet(file_path, geolocator):
        # file_path is absolute path of csv file containing addresses of insterests
        # Assuming csv file, and first row of file is column header and second row is comment

        # data_dir = os.path.dirname(file_path)       # assuming csv file is in a data folder
        df = pd.read_csv(file_path, header=0)

        # Define a function to determine if a row is a comment
        def is_comment(row):
            # Example condition: check if 'ADDRESS' or 'CITY' is NaN (or another identifier for comments)
            if pd.isna(row['ADDRESS']) or pd.isna(row['CITY']):
                return True
            # Additional checks can be added here if there are other patterns indicating comments
            return False

        # Apply the function to each row to determine if it's a comment
        df['is_comment'] = df.apply(is_comment, axis=1)

        # Filter out the rows that are potentially comments
        filtered_df = df[~df['is_comment']]
        filtered_df = filtered_df.drop(columns=['is_comment'])
        failed_indice = []
        for index, row in filtered_df.iterrows():
            try:
                # Extract and check if address, city, state, and zipCode are NaN
                address = row['ADDRESS'] if pd.notna(row['ADDRESS']) else None
                city = row['CITY'] if pd.notna(row['CITY']) else None
                state = row['STATE OR PROVINCE'] if pd.notna(row['STATE OR PROVINCE']) else None
                zipCode = row['ZIP OR POSTAL CODE'] if pd.notna(row['ZIP OR POSTAL CODE']) else None
                
                # Optionally get latitude and longitude
                latitude = row['LATITUDE'] if 'LATITUDE' in row and pd.notna(row['LATITUDE']) else None
                longitude = row['LONGITUDE'] if 'LONGITUDE' in row and pd.notna(row['LONGITUDE']) else None

                # Check for the necessary location information
                if pd.isna(address) and (pd.isna(latitude) or pd.isna(longitude)):
                    print(f"Missing address and coordinates at index {index}")
                    failed_indice.append(index)
                    continue
                if pd.isna(city) or pd.isna(state) or pd.isna(zipCode):
                    print(f"Missing city/state/zip information at index {index}")
                    failed_indice.append(index)
                    continue

                # Use address if available; otherwise, rely on latitude and longitude
                if pd.notna(address):
                    # Construct full address, skipping any empty parts
                    full_address = ' '.join(filter(None, [address, city, state, str(int(zipCode))]))
                    location = geolocator.geocode(full_address)
                    if location:
                        latitude = location.latitude
                        longitude = location.longitude
                    else:
                        print(f"Coordinates not found for the address: {full_address}")
                        failed_indice.append(index)
                        continue
                else:
                    if latitude is not None and longitude is not None:
                        full_address = f"lat{latitude:.6f}_long{longitude:.6f}"
                    else:
                        print(f"Insufficient data at index {index} to proceed with geocoding.")
                        failed_indice.append(index)
                        continue
                
                # Prepare the filename-friendly address name
                address_name = full_address.replace('/', '-').replace(' ', '_')
                request_and_save(address_name, latitude, longitude, save_dir=data_dir)

            except KeyError as e:
                print(f"Missing data for key: {e} at row index: {index}")
            except Exception as e:
                print(f"Error processing row index {index}: {e}")

    def processCoordinates(coord_list, save_dir):
        for latitude, longitude in coord_list:
            location_name = f"lat{latitude:.6f}_long{longitude:.6f}"
            request_and_save(location_name, latitude, longitude, save_dir)
    
    failed_fname = "Failed_Google_Solar_Locations" # name for failed locations file. Do not add file type at the end
    # Alters failed addresses document title to be unique and not overwrite another file
    # FIXME: data_dir needs to be passed in or initialized
    failed_fpath = get_unique_filename(data_dir, failed_fname, '.txt')

    # TODO: make data query based on src input type

    if isinstance(src, list):      # input is a list of coordinates
        coord_list = src
        # FIXME: define save directory if processing a list of coordinates
        # processCoordinates(coord_list, save_dir=?)
    else:           # input is csv file
        geolocator = GoogleV3(api_key)
        fnames = [fname for fname in os.listdir(data_dir) if fname.endswith('.csv')]
        for fname in fnames:
            f_path = os.path.join(data_dir, fname)
            processSpreadsheet(f_path, geolocator)

def test1():
    # Access the API key
    load_dotenv()  # This loads the variables from .env
    api_key = os.getenv('GOOGLE_SOLAR_API_KEY')
    if api_key:
        print("API Key found:", api_key)
    else:
        print("API Key not found. Please check your .env file")
    csv_path = '/home/psc/Desktop/solarRec/data/test_data2/TestAddresses.csv'
    data_dir = os.path.dirname(csv_path)
    query_from_API(csv_path, data_dir,
                   api_key=api_key)

def main():
    test1()

if __name__ == "__main__":
    main()