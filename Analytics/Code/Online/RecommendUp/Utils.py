import numpy as np

class color:

   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


class airportFeatures:

   CENTER_LAT = 37.363589 
   CENTER_LON = -121.929066

   
class geoParams:

   EARTH_RADIUS = 6378137 # in meters
   EQUATOR_CIRCUMFERENCE = 2 * np.pi * EARTH_RADIUS
   #INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
   #ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0
