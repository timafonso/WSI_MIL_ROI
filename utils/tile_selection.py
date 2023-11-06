from utils.gdc_api_utils import getTile
from utils.preprocessing_utils import getTissuePixelCoords

import math
import numpy as np

THUMBNAIL_LEVEL = 9
TILE_DIMENSTIONS = 512


def getNumberOfTiles(img_dim, magnification_lvl, tile_dim=TILE_DIMENSTIONS):
    return (math.floor(img_dim[0] * 2**(magnification_lvl - THUMBNAIL_LEVEL) / tile_dim),
            math.floor(img_dim[1] * 2**(magnification_lvl - THUMBNAIL_LEVEL) / tile_dim))


def getTilesFromCoords(coords, magnification_lvl, init_mag_lvl):
    number_of_levels = magnification_lvl - init_mag_lvl
    final_coords = []
    for coord in coords:
        offset = pow(2, number_of_levels)
        initial_x = coord[0] * offset
        initial_y = coord[1] * offset
        for x in range(initial_x, initial_x + offset, 1):
            for y in range(initial_y, initial_y + offset, 1):
                final_coords.append((x,y))
    

def getTileCoords(pixel_coords, img_dim, magnification_lvl, init_mag_lvl=THUMBNAIL_LEVEL, tile_dim=TILE_DIMENSTIONS):
    """
    Given the coordinates of a pixel in the thumbnail of the slide, it returns the corresponding patch coords at a certain magnification

    parameters:
    -----------
    pixel_coords: tuple (x,y) with coords of the pixel
    img_dim: tuple (dim_x, dim_y) with the dimensions of the thumbnail
    amp_lvl: amplification/magnification level
    tile_dim: dimension of the patches (default is 512)

    returns:
    --------
    Tuple (x_coord, y_coord)
    """
    number_of_levels = magnification_lvl - init_mag_lvl
    possible_tiles = getNumberOfTiles(img_dim, magnification_lvl, tile_dim) 
    tile_coords = (pixel_coords[0] * 2**(number_of_levels), pixel_coords[1] * 2**(number_of_levels))
    tile_coords = (min(math.floor(tile_coords[0]/tile_dim), possible_tiles[0]), 
                   min(math.floor(tile_coords[1]/tile_dim), possible_tiles[1]))

    return tile_coords


def getTissueTileCoords(init_img, amp_lvl, init_amp_lvl, tile_dim=TILE_DIMENSTIONS):
    """
    Given the thumbnail img of a WSI, it returns, for a amplification/magnification level, the coords of the tiles where there is
    a considerable amount of tissue

    parameters:
    -----------
    img (Image): thumbnail image of WSI
    amp_lvl (int): amplification/magnification level

    returns:
    --------
    List of tuples (x_coord, y_coord)
    """

    tissue_coords = []
    tissue_pixels = []
    init_img = np.array(init_img).transpose(1,0,2)
    tissue_pixels, im_bw = getTissuePixelCoords(init_img)

    for coords in tissue_pixels:
        x, y = coords[0], coords[1]
        pixel_color = init_img[x,y]
        patch_coords = getTileCoords((x,y), (init_img.shape[0], init_img.shape[1]), amp_lvl, init_amp_lvl, tile_dim)
        if patch_coords not in tissue_coords:
            tissue_coords.append(patch_coords)
        np.multiply(pixel_color, [0.8, 0.2, 0.2], out=pixel_color, casting="unsafe")

    return tissue_coords, im_bw

def getImageTiles(slide_id, amp_lvl, init_amp_lvl=THUMBNAIL_LEVEL):
    """
    Given the id of a WSI and an amplification level, it returns the coords of the tiles where there is a considerable 
    amount of tissue

    parameters:
    -----------
    slide_id (str): id of the WSI
    amp_lvl (int): amplification/magnification level

    returns:
    --------
    List of tuples (x_coord, y_coord)
    """
    tiles = []
    img = getTile(slide_id, init_amp_lvl, 0, 0)
    tiles, b_w = getTissueTileCoords(img, amp_lvl, init_amp_lvl)
    return tiles, b_w