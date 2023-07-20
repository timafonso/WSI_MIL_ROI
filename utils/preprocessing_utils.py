from math import sqrt
from PIL import Image
import numpy as np
import cv2


TISSUE_THRESHOLD = 2
ARTIFACT_THRESHOLD = 0 
COLOR_DISTANCE_THRESHOLD = 35


def getTissuePixelCoords(img):
    tissue_coords = []

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, im_bw = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel33 = np.ones((3, 3), np.uint8)
    im_bw = cv2.erode(im_bw, kernel33)
    im_bw = cv2.dilate(im_bw, kernel33)

    black_pixel_coords = np.column_stack(np.where(im_bw == 0))

    avg_color = calculateAvgColor(img, black_pixel_coords)
    
    for pixel_index in range(black_pixel_coords.shape[0]):
        x, y = black_pixel_coords[pixel_index, 0], black_pixel_coords[pixel_index, 1] 
        if calculateEuclideanDistance(avg_color, img[x, y]) < COLOR_DISTANCE_THRESHOLD:
            tissue_coords.append((x, y))

    return tissue_coords, im_bw


def getTissuePercentage(img):
    cv2_img = np.array(img)

    cv2_img = cv2_img[:,:,::-1].copy()
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])

    mask = cv2.inRange(cv2_img, lower_white, upper_white)
   

    result = cv2.bitwise_and(cv2_img, cv2_img, mask=mask)

    kernel = np.ones((2, 2), np.uint8)
    result = cv2.dilate(result, kernel)

    black_percent = countBlackPixels(result)
    return black_percent*100


def countBlackPixels(img):
    cv2_img = np.array(img)
    black_pixels = 0
    total_pixels = 0
    for x in range(cv2_img.shape[0]):
        for y in range(cv2_img.shape[1]):
            rgb = cv2_img[x, y]
            if np.array_equal(rgb, np.array([0,0,0])):
                black_pixels += 1
            total_pixels += 1
    return black_pixels/total_pixels


def getBackgroundColor(img):
    cv2_img = np.array(img)

    cv2_img = cv2_img[:,:,::-1].copy()
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])

    mask = cv2.inRange(cv2_img, lower_white, upper_white)
    result = cv2.bitwise_and(cv2_img, cv2_img, mask=mask)

    avg_red = 0
    avg_green = 0
    avg_blue = 0
    total_pixels = 0
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            rgb = result[x, y]
            if not np.array_equal(rgb, np.array([0,0,0])):
                avg_red += rgb[0]
                avg_green += rgb[1]
                avg_blue += rgb[2]
                
                total_pixels += 1

    avg_red /= total_pixels
    avg_green /= total_pixels
    avg_blue /= total_pixels

    return (int(avg_red), int(avg_green), int(avg_blue))


def getArtifactPercentage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_range = np.array([72,130,6])
    upper_range = np.array([125, 266, 140])
    mask = cv2.inRange(img, lower_range, upper_range)
    total_pixels = img.shape[0] * img.shape[1]
    artifact_pixels = cv2.countNonZero(mask)
    artifact_percentage = (artifact_pixels / total_pixels) * 100

    return artifact_percentage


def addMargin(img, top, right, bottom, left, color):
    width, height = img.size
    new_width = width + right + left
    new_height = height + top + bottom   
    result = Image.new(img.mode, (new_width, new_height), color)
    result.paste(img, (left, top))
    return result

def calculateAvgColor(img, pixel_coords):
    """Calculates the average color of a group of pixels in an image

    Args:
        img (np array): image
        pixel_coords (list<tuple<int>>): pixel coordinates

    Returns:
        float: average color
    """
    total_r, total_g, total_b = 0, 0, 0
    for coord in pixel_coords:
        pixel_color = img[coord[0], coord[1]]
        total_r += pixel_color[0]
        total_b += pixel_color[1]
        total_g += pixel_color[2]
    
    total_r /= len(pixel_coords)
    total_b /= len(pixel_coords)
    total_g /= len(pixel_coords)

    return (total_r, total_b, total_g)

def calculateEuclideanDistance(color1, color2):
    """calculates the Euclidian Distance between 2 colors (in RGB)

    Args:
        color1 (tuple<int>)
        color2 (tuple<int>)

    Returns:
        float: euclidian distance between color1 and color2
    """
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    return sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)
