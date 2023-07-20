from utils.gdc_api_utils import getTile
from utils.preprocessing_utils import getTissuePercentage, getBackgroundColor, addMargin, TISSUE_THRESHOLD
from utils.tile_selection import getImageTiles
from utils.file_utils import create_dir, is_file, save_img

from os import listdir
from math import log2
from PIL import ImageFile
from random import seed
from shutil import rmtree
from threading import Thread

import pandas as pd


# seed random number generator
seed(1)
ImageFile.LOAD_TRUNCATED_IMAGES = True


NUM_THREADS = 4

#============================================ Download Patches =====================================================

      
def downloadWSIPatch(folder, slide_id, slide_label, magnification, magnification_lvl, coords, start_index, end_index, num_threads=NUM_THREADS):
    """fetches a WSI patch at a certain magnification and coords and stores it in a file under the dir
    bags/bag{slide_id} with the na with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(downloadWSIPatch, slide_ids[i], slide_labels[i], magnification, coord[0], coord[1], test) 
                       for coord in tissue_coords]me patch_{coord_x}_{coord_y}

    Args:
        folder (str): folder where patches are to be stored
        slide_id (str): id of the slide in the GDC API
        magnification (int): magnification level of the patch (9-15 for 512x512 images)
        coord_x (int): coord x of the patch
        coord_y (int): coord y of the patch
    """

    for i in range(start_index, end_index, num_threads):
        coord_x = coords[i][0]
        coord_y = coords[i][1]
        file_path = "{main_dir}/bag_{label}_{bag}/patch_{x}_{y}.png".format(
            main_dir=folder, 
            mag=magnification, 
            bag=slide_id, 
            label=slide_label, 
            x=coord_x, 
            y=coord_y)
        
        if not is_file(file_path): 
            img = getTile(slide_id, magnification_lvl, coord_x, coord_y)
            right_padding = 512 - img.size[0] if img.size[0] < 512 else 0
            bottom_padding = 512 - img.size[1] if img.size[1] < 512 else 0
            background_color = getBackgroundColor(img)
            img = addMargin(img, 0, right_padding, bottom_padding, 0, background_color)

            if img.size[0] >= 512 and img.size[1] >= 512 and getTissuePercentage(img) > TISSUE_THRESHOLD:
                img = img.crop((0, 0, 512, 512))
                save_img(img, file_path)


def downloadPatchesFromFile(csv_filename, folder_name, number_of_slides, magnification, label, num_threads=4):
    """Downloads patches with the GDC API indicated by the slide id in a csv file.
    The csv file level magnification corresponds to 5x magnification.
    It is assumed that the all the slide ids in the csv file correspond to either positive or negative slides

    Args:
        csv_filename (str): name of csv file indicating slide_id and the deepzoom level for 5x magnification
        folder_name (str): folder to store the patches
        number_of_slides (int): number of slides to download
        magnification (int): magnification desired
        label (int): label of the tiles (0 or 1)
        num_threads (int, optional): number of threads. Defaults to 4.
    """
    slide_info = pd.read_csv(csv_filename)
    create_dir("{main_dir}".format(main_dir=folder_name))
    for i in range(len(slide_info)):
        print(i)
        if create_dir("{main_dir}/bag_{label}_{id}".format(main_dir=folder_name, label=label, id=slide_info["id"][i])):
            max_magnification = 10/slide_info["mpp"][i]
            levels_to_decrease = int(log2(max_magnification / magnification))
            magnification_level = slide_info["max_level"][i] - levels_to_decrease
            tissue_coords, _ = getImageTiles(slide_info["id"][i], magnification_level)
            threads = []
            for thread_index in range(num_threads):
                start_index = thread_index
                end_index = len(tissue_coords)
                t = Thread(
                    target=downloadWSIPatch, args=(
                    folder_name,
                    slide_info["id"][i], 
                    label, 
                    magnification, 
                    magnification_level, 
                    tissue_coords, 
                    start_index, 
                    end_index)
                )
                threads.append(t)
            
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            number_of_slides -= 1
            if number_of_slides == 0:
                break
            print("number", number_of_slides)

        else:
            print("exists")
            number_of_slides -= 1
            print("number", number_of_slides)
            if number_of_slides == 0:
                break

    for bag_name in listdir(folder_name):
        if len(listdir(folder_name+"/"+bag_name)) < 5:
            print("deleted_dir with", len(listdir(folder_name+"/"+bag_name)), "images")
            rmtree(folder_name+"/"+bag_name, ignore_errors=False, onerror=None)