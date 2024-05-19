from math import pi, ceil, cos


# Calculates the meter to degree longitude conversion scalar using the latitude
# Required because Earth is not a perfect sphere, so difference in longitude relies on latitude
def meter_to_long(latitude):
    return 1 / ((pi / 180) * 6378137 * cos(latitude * pi / 180))

# Given an origin (latitude, longitude) coordinate and a height and width integer, will output an array of 
# (latitude, longitude) coordinates to make a rectangular grid of images in an area without overlap.
# radius (default 70) is the radius in meters of the coverage area for one google solar API image.
def generate_grid(origin_coord, height, width, RADIUS=70):
    # Fixes non integer paramters
    height = ceil(height)
    width = ceil(width)
    meter_to_lat = 1 / ((pi / 180) * 6378000)
    # Converts origin coord to top left box's coord
    starting_lat = float(origin_coord[0]) + (height/2 - 0.5) * meter_to_lat * RADIUS * 2
    starting_long = float(origin_coord[1]) - (width/2 - 0.5) * meter_to_long(starting_lat) * RADIUS * 2
    grid_coords = []
    for i in range (0, height):
        for j in range (0, width):
            lat = starting_lat - i * meter_to_lat * RADIUS * 2
            long = starting_long + j * meter_to_long(lat) * RADIUS * 2
            grid_coords.append((lat, long))
    return grid_coords