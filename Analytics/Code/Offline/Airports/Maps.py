import Image, urllib, StringIO
import time
import numpy as np
import sys
from math import log, exp, tan, atan, pi, ceil

# Static parameters

EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0

def lat_lon_to_pixels(lat, lon, zoom):

    mx = (lon * ORIGIN_SHIFT) / 180.0
    my = log(tan((90 + lat) * pi/360.0))/(pi/180.0)
    my = (my * ORIGIN_SHIFT) /180.0
    res = INITIAL_RESOLUTION / (2**zoom)
    px = (mx + ORIGIN_SHIFT) / res
    py = (my + ORIGIN_SHIFT) / res

    return px, py


def pixels_to_lat_lon(px, py, zoom):

    res = INITIAL_RESOLUTION / (2**zoom)
    mx = px * res - ORIGIN_SHIFT
    my = py * res - ORIGIN_SHIFT
    myLat = (my / ORIGIN_SHIFT) * 180.0
    myLat = 180 / pi * ( 2*atan( exp(myLat*pi/180.0) ) - pi/2.0 )
    myLon = (mx / ORIGIN_SHIFT) * 180.0

    return myLat, myLon


############################################

# a neighbourhood in Lajeado, Brazil:

#upperleft =  '-29.44,-52.0'  
#lowerright = '-29.45,-51.98'

upperleft = '51.47,-0.5'
lowerright = '51.46,-0.48'

zoom = 16   # be careful not to get too many images!

ullat, ullon = map(float, upperleft.split(','))
lrlat, lrlon = map(float, lowerright.split(','))

############################################

def SaveMap( myImageObject, myFilename ):

    #myImg   = myImageObject.convert('L')
    #myImage = np.array(myImg)

    #fft_mag = np.abs( np.fft.fftshift(np.fft.fft2(myImg)) )

    #visual = np.log(fft_mag)
    #visual = (visual - visual.min()) / (visual.max() - visual.min())

    #myResult = Image.fromarray((visual * 255).astype(np.uint8))
    myImageObject.save( myFilename )


def DrawMap( myCenterLat, myCenterLon, myResoLat, myResoLon, myZoom, myFilename ):

    # Set some important parameters
    scale = 1
    maxsize = 640

    # UL = upper left, LR = lower right
    myLat_UL = myCenterLat + 0.5 * myResoLat
    myLon_UL = myCenterLon - 0.5 * myResoLon
    myLat_LR = myCenterLat - 0.5 * myResoLat
    myLon_LR = myCenterLon + 0.5 * myResoLon

    #myLat_UL = ullat
    #myLon_UL = ullon
    #myLat_LR = lrlat
    #myLon_LR = lrlon

    # convert all these coordinates to pixels
    ulx, uly = lat_lon_to_pixels(myLat_UL, myLon_UL, myZoom)
    lrx, lry = lat_lon_to_pixels(myLat_LR, myLon_LR, myZoom)

    print "-->", ulx, uly
    print "-->", lrx, lry

    # calculate total pixel dimensions of final image
    dx, dy = lrx - ulx, uly - lry
    
    # calculate rows and columns
    cols, rows = int( ceil(dx/maxsize)), int( ceil(dy/maxsize) )
    print "Columns, Rows = %d, %d" % (cols, rows)

    # calculate pixel dimensions of each small image
    bottom     = 120
    largura    = int( ceil(dx/cols) )
    altura     = int( ceil(dy/rows) )
    alturaplus = altura + bottom

    final = Image.new("RGB", (int(dx), int(dy)))
    
    for x in range(cols):
        for y in range(rows):

            dxn = largura * (0.5 + x)
            dyn = altura * (0.5 + y)
            latn, lonn = pixels_to_lat_lon(ulx + dxn, uly - dyn - bottom/2, zoom)
            position = ','.join((str(latn), str(lonn)))
            print x, y, position
            urlparams = urllib.urlencode({'center': position,
                                          'zoom': str(zoom),
                                          'size': '%dx%d' % (largura, alturaplus),
                                          'maptype': 'map',
                                          'sensor': 'false',
                                          'scale': scale})
            url = 'http://maps.google.com/maps/api/staticmap?' + urlparams
            f=urllib.urlopen(url)
            myImage = Image.open(StringIO.StringIO(f.read()))
            final.paste( myImage, (int(x*largura), int(y*altura)) )

    final.show()
    SaveMap( myImage, myFilename )
