#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pynoramio as pymap
import csv
import Maps

from googleplaces import GooglePlaces, types, lang

#YOUR_API_KEY = 'AIzaSyBoggHOZ8jWDzO69jVGNkoX4F-m8FYKa0Q'
YOUR_API_KEY = 'AIzaSyDo-xr-5RnuSbHDPJGRpR-YCG_cVQ9CXiI'
#print YOUR_API_KEY
#print type(YOUR_API_KEY)
#assert(0)
google_places = GooglePlaces(YOUR_API_KEY)

def readAirportList( myFilename = './Data/Airports.csv' ):

    myAirportData = pd.read_csv( myFilename, sep = ',' )
    return myAirportData.columns.values.tolist(), myAirportData.values.tolist()

    
# You may prefer to use the text_search API, instead.
myQueryResult = google_places.nearby_search(
                keyword = 'Airport',
                location = 'London, England',
                radius = 50000 )
                #types = [types.TYPE_AIRPORT])

#print myQueryResult.__dict__

for myPlace in myQueryResult._places :

    print myPlace.__dict__
    print myPlace._query_instance
    myLat = myPlace._geo_location['lat']
    myLng = myPlace._geo_location['lng']
    
    print myLat, myLng
    #myLatMin = myLat + 0.1
    #myLngMin = myLng - 0.1
    #myLatMax = myLat - 0.1
    #myLngMax = myLng + 0.1
    #print myLatMin, myLatMax
    #print myLngMin, myLngMax

    myLat = myLat
    myLng = myLng

    myFilename = myPlace._name.replace(" ","_")
    myFilename = "./" + myFilename + ".bmp"

    Maps.DrawMap( myLat, myLng, 0.01, 0.02, 16, myFilename )

    #myPyMap = pymap.Pynoramio()

    #myMap = myPyMap.get_from_area(myLatMin, myLngMin, myLatMax, myLngMax, picture_size = None)
    #print type(myMap)
    #print myMap

if myQueryResult.has_attributions:
        print myQueryResult.html_attributions

assert(0)

myHeaders, myAirportData = readAirportList()
print myHeaders

code_index = myHeaders.index( "'AirportCode'" )
lat_index  = myHeaders.index( "'Latitude'" )
lng_index  = myHeaders.index( "'Longitude'" )  

for myAirport in myAirportData:

    myAirportCode = myAirport[code_index]
    myLatitude    = myAirport[lat_index]
    myLongitude   = myAirport[lng_index]

    print myLatitude
    print myLongitude

    myQueryResult = google_places.nearby_search( name = 'Airport',
                                                 sensor = False,
                                                 keyword = 'Airport',
                                                 lat_lng = { 'lat' : myLatitude, 'lng' : myLongitude } ,
                                                 radius  = 2000,
                                                 types = [ types.TYPE_AIRPORT ] 
                                                 ) 

    if myQueryResult.has_attributions:
        print myQueryResult.html_attributions

'''
for place in query_result.places:
    # Returned places from a query are place summaries.
    print place.name
    print place.geo_location
    print place.place_id

    # The following method has to make a further API call.
    place.get_details()
    # Referencing any of the attributes below, prior to making a call to
    # get_details() will raise a googleplaces.GooglePlacesAttributeError.
    print place.details # A dict matching the JSON response from Google.
    print place.local_phone_number
    print place.international_phone_number
    print place.website
    print place.url

    # Getting place photos

    for photo in place.photos:
        # 'maxheight' or 'maxwidth' is required
        photo.get(maxheight=500, maxwidth=500)
        # MIME-type, e.g. 'image/jpeg'
        photo.mimetype
        # Image URL
        photo.url
        # Original filename (optional)
        photo.filename
        # Raw image data
        photo.data
'''
