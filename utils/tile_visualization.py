import numpy as np
from matplotlib import pyplot as plt
from utils.gdc_api_utils import getTile

import math

def getPixelsInThumbnail(slide_id, magnification_level, coords, show_original=False):
    img = getTile(slide_id, 9, 0, 0)
    img = np.array(img).transpose((1,0,2))
    
    if show_original:
        original_img = img.transpose((1,0,2)).copy()
        
    levels = magnification_level - 9 
    num_tiles = 2**levels

    pixels_per_patch = int(512/num_tiles)

    for index in range(len(coords)):
        pixel_x = int(coords[index][0] * pixels_per_patch)
        pixel_y = int(coords[index][1] * pixels_per_patch)
        pixels_to_color = img[pixel_x:pixel_x+pixels_per_patch, pixel_y:pixel_y+pixels_per_patch, :]
        np.multiply(pixels_to_color,  [0.8,0.2,0.2], out=pixels_to_color, casting="unsafe")

    img = img.transpose((1,0,2))

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(img)
    if show_original:
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(original_img)
    plt.show()