from utils.file_utils import create_dir, is_file, save_img
from utils.gdc_api_utils import getTile
from utils.preprocessing_utils import getTissuePercentage, getBackgroundColor, addMargin, TISSUE_THRESHOLD
from utils.tile_selection import getImageTiles
<<<<<<< HEAD
from utils.data_augmentations import normal_transforms, aug_transforms
=======
from utils.tile_visualization import getPixelsInThumbnail
>>>>>>> 87e3e989f354e1bf459b2ad1436dcea0fd7bc7e8

from KimiaNet.KimiaNet_PyTorch_Feature_Extraction import model_final as feature_extractor

import torch
import os
import h5py
from os import listdir
<<<<<<< HEAD
from math import log2, ceil
from PIL import Image, ImageChops
=======
from math import log2
from PIL import Image
>>>>>>> 87e3e989f354e1bf459b2ad1436dcea0fd7bc7e8
from PIL import ImageFile
from random import seed
from shutil import rmtree
from threading import Thread, Lock
from multiprocessing import Process, Manager

import numpy as np
import pandas as pd
import random

lock = Lock()

# seed random number generator
seed(1)
ImageFile.LOAD_TRUNCATED_IMAGES = True

NUM_THREADS = 4

<<<<<<< HEAD
#============================================ Download Patches Locally =====================================================
=======
#============================================ Download Patches =====================================================

      
>>>>>>> 87e3e989f354e1bf459b2ad1436dcea0fd7bc7e8
def downloadWSIPatch(folder, slide_id, slide_label, percentage, magnification_lvl, coords, start_index, end_index, num_threads=NUM_THREADS):
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
        file_path = "{main_dir}/{id}_{label}_{perc}/patch_{x}_{y}.png".format(
            main_dir=folder,  
            id=slide_id, 
            label=slide_label, 
            perc=percentage,
            x=coord_x, 
            y=coord_y)
        
        if not is_file(file_path): 
            img = getTile(slide_id, magnification_lvl, coord_x, coord_y)
            if img.size[0] < 512 or img.size[1] < 512:
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
        if create_dir("{main_dir}/{id}_{label}_{perc}".format(main_dir=folder_name, label=label, id=slide_info["id"][i], perc=slide_info["tumor_percentage"][i])):
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
                    slide_info["tumor_percentage"][i], 
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

            img, _ = getPixelsInThumbnail(slide_info["id"][i], magnification_level, tissue_coords)
            save_img(Image.fromarray(img), "{main_dir}/{id}_{label}_{perc}/thumbnail.png".format(
                        main_dir=folder_name,  
                        id=slide_info["id"][i], 
                        label=label, 
                        perc=slide_info["tumor_percentage"][i]
                        ))

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

# =================================== Create Embeddings from Local Images ===============================================

def createBag(path, bag_name, tiles, slide_id, case_id, label, hdf5_file, batch_size=64, num_threads=4):
    number_of_tiles = len(tiles)
    if number_of_tiles == 0:
        print("0 tiles, ", slide_id)
        return
    batch_iterations = ceil(number_of_tiles/ batch_size)
    print(slide_id)

    complete_embeddings = []
    complete_coords = []

    file = h5py.File(hdf5_file, "r")

    if slide_id in file:
        print("skipping ", slide_id)
        return
    file.close()

    del file

    for batch_index in range(batch_iterations):
        lower_limit = batch_index*batch_size
        upper_limit = (batch_index+1)*batch_size
        if upper_limit > number_of_tiles:
            upper_limit = number_of_tiles
        
        batch_tiles = tiles[lower_limit : upper_limit]
        batch_tiles_size = len(batch_tiles)
        images = [None] * batch_tiles_size
        coords = [None] * batch_tiles_size

        threads = []
        for thread_index in range(num_threads):
            start_index = thread_index
            end_index = batch_tiles_size

            t = Thread(
                    target=createPatch, args=(
                    path, 
                    bag_name,
                    batch_tiles, 
                    images,
                    coords,
                    start_index,
                    end_index,
                    num_threads)
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        del threads

        transformed_images = applyImageTransforms(images)

        complete_coords += coords   
        
        del images, coords

        embeddings = createEmbeddings(transformed_images)
        complete_embeddings.append(embeddings)
        del transformed_images, embeddings
        
    final_embeddings = torch.cat(complete_embeddings, dim=1)

    del complete_embeddings

    with h5py.File(hdf5_file, "a", libver='latest', swmr=True, rdcc_nbytes=1024*3000) as file:
        group = file.create_group(slide_id)
        group.create_dataset('slide_id', data=slide_id)
        group.create_dataset('case_id', data=case_id)
        group.create_dataset('coords', data=np.array(complete_coords))
        embeddings_shape = final_embeddings.shape
        embeddings_dataset = group.create_dataset('embeddings', shape=embeddings_shape, dtype='float32', chunks=True)
        embeddings_dataset[:] = np.array(final_embeddings)
        group.create_dataset('label' , data=np.array(label))
    
    del slide_id, case_id, final_embeddings, label, complete_coords


def applyImageTransforms(images):
    transformed_imgs = []
    transformed_imgs.append(torch.stack([normal_transforms(img) for img in images], 0))
    transformed_imgs.append(torch.stack([aug_transforms[0](img) for img in images], 0))
    transformed_imgs.append(torch.stack([aug_transforms[1](img) for img in images], 0))
    return torch.stack(transformed_imgs, dim=0)

def createEmbeddings(images):
    embeddings = []
    for img_index in range(images.shape[0]):
        image = images[img_index, :, :, :, :]
        if image.shape[0] != 1:
            image = image.squeeze(0)
        embedding = feature_extractor(image)[0]
        embeddings.append(embedding)
        del embedding
    
    return torch.stack(embeddings)

def createPatch(path, bag_name, tiles, images, coords, start, end, num_threads):
    for i in range(start, end, num_threads):
        tile_coords = tiles[i].split(".")[0].split("_")[1:]
        coord = (int(tile_coords[0]), int(tile_coords[1]))
        coords[i] = coord
        img = Image.open("{}/{}/{}".format(path, bag_name, tiles[i])) 
        img = img.crop((0,0,512,512))
        images[i]  = np.array(img)
        

# =================================== Create Embeddings Directly ========================================================

def createBagsFromFolder(slide_data, hdf5_file, label_tag, magnification, num=0, sampling=False, coords_file=None):
    df = pd.read_csv(slide_data)
    
    if coords_file != None:
        with open(coords_file, 'rb') as f:
            coords = np.load(f, allow_pickle=True)
            slides = list(coords.item().keys())
            df = df[df['slide_id'].isin(slides)]
    createBags(df, hdf5_file, label_tag, magnification, num, sampling, coords_file)

        
def createBags(df, hdf5_file, label_tag, magnification=5, num=0, sampling=False, coords_file=None):
    count_pos, count_neg = 0, 0
    initial_magnification = 10
    c = 0
    if coords_file != None:
        with open(coords_file, 'rb',) as f:
            coords = np.load(f, allow_pickle=True)
    for index, slide in df.iterrows():
        c += 1
        slide_id = slide["slide_id"]
        label = int(slide[label_tag])

        if index < num:
            print("skip", slide_id)
            print(count_pos, count_neg)
            continue
        
        if os.path.isfile(hdf5_file):
            with h5py.File(hdf5_file) as file:
                if slide_id in file:
                    print("skipping ", slide_id)
                    continue

        case_id = slide["case_id"]
        max_magnification = 10/slide["mpp"]    

        levels_to_decrease = int(log2(max_magnification / magnification))
        magnification_level = slide["max_level"] - levels_to_decrease   

        if sampling:
            if coords_file != None:
                initial_levels_to_decrease = int(log2(max_magnification / initial_magnification))
                initial_magnification_level = slide["max_level"] - initial_levels_to_decrease
                slide_coords = coords.item().get(slide["slide_id"])
                level_offset = 2**(magnification_level - initial_magnification_level)
                tissue_coords = []
                for coord in slide_coords:
                    for i in range(level_offset):
                        for j in range(level_offset):
                            tissue_coords.append(np.array([level_offset*coord[0]+i,level_offset*coord[1]+j]))
                tissue_coords = np.array(tissue_coords)
                print("tissue_coords", tissue_coords.shape)
            else:  
                tissue_coords, _ = getImageTiles(slide_id, magnification_level)
                len_tissue_coords = len(tissue_coords)
                if len_tissue_coords > 300:
                    sample_number = int(0.6*len_tissue_coords)
                    tissue_coords = random.sample(tissue_coords, sample_number)
                tissue_coords = np.array(tissue_coords)
        else:
            tissue_coords, _ = getImageTiles(slide_id, magnification_level)
            
        createBag(hdf5_file, slide_id, magnification_level, case_id, tissue_coords, label)
        del slide_id, tissue_coords, case_id, label, magnification_level, levels_to_decrease, max_magnification


def createBag(hdf5_file, slide_id, magnification_level, case_id, tissue_coords, label, batch_size=64, num_threads=4):
    number_of_tiles = tissue_coords.shape[0]
    if number_of_tiles == 0:
        return
    
    batch_iterations = ceil(number_of_tiles/ batch_size)

    complete_embeddings = []
    complete_coords = []

    for batch_index in range(batch_iterations):
        lower_limit = batch_index*batch_size
        upper_limit = (batch_index+1)*batch_size
        if upper_limit > number_of_tiles:
            upper_limit = number_of_tiles
        
        batch_tiles = tissue_coords[lower_limit : upper_limit]
        batch_tiles_size = len(batch_tiles)
        images = [None] * batch_tiles_size

        threads = []
        for thread_index in range(num_threads):
            start_index = thread_index
            end_index = batch_tiles_size

            t = Thread(
                    target=fetch_patch, args=(
                    slide_id, 
                    magnification_level,
                    batch_tiles,
                    images,
                    start_index,
                    end_index,
                    num_threads)
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        del threads

        final_images = [None] * batch_tiles_size
        final_coords = [None] * batch_tiles_size
           
        for i in range(0, batch_tiles_size, 1):
            img = images[i]
            if img == None:
                continue
            if img.size[0] < 512 or img.size[1] < 512:
                right_padding = 512 - img.size[0] if img.size[0] < 512 else 0
                bottom_padding = 512 - img.size[1] if img.size[1] < 512 else 0
                background_color = getBackgroundColor(img)
                img = addMargin(img, 0, right_padding, bottom_padding, 0, background_color)
                del right_padding, bottom_padding, background_color

            if img.size[0] >= 512 and img.size[1] >= 512 and getTissuePercentage(img) > TISSUE_THRESHOLD:
                img = img.crop((0, 0, 512, 512))
                final_images[i] = np.array(img)
                final_coords[i] = batch_tiles[i]
            del img

        transformed_images = applyImageTransforms([i for i in list(final_images) if i is not None])

        complete_coords += [i for i in list(final_coords) if i is not None]   
        del final_images, final_coords
        if len(transformed_images) > 0:
            complete_embeddings.append(createEmbeddings(transformed_images))
        del transformed_images

    
    if len(complete_embeddings) > 0:
        final_embeddings = torch.cat(complete_embeddings, dim=1)
        del complete_embeddings
        print(final_embeddings.shape)
        if final_embeddings.shape[1] < 30:
            return False

        with h5py.File(hdf5_file, "a", libver='latest', swmr=True, rdcc_nbytes=1024*3000) as file:
            group = file.create_group(slide_id)
            group.create_dataset('slide_id', data=slide_id)
            group.create_dataset('case_id', data=case_id)
            group.create_dataset('coords', data=np.array(complete_coords))
            embeddings_shape = final_embeddings.shape
            embeddings_dataset = group.create_dataset('embeddings', shape=embeddings_shape, dtype='float32', chunks=True)
            embeddings_dataset[:] = np.array(final_embeddings.cpu())
            group.create_dataset('label', data=np.array(label))

        
        del slide_id, case_id, final_embeddings, label, complete_coords
        return True
    return False

def applyImageTransforms(images):
    transformed_imgs = []
    if len(images) > 0:
        transformed_imgs.append(torch.stack([normal_transforms(img) for img in images], 0).detach())
        transformed_imgs.append(torch.stack([aug_transforms[0](img) for img in images], 0).detach())
        transformed_imgs.append(torch.stack([aug_transforms[1](img) for img in images], 0).detach())
        return torch.stack(transformed_imgs, dim=0).detach()
    else:
        return []

def createEmbeddings(images):
    embeddings = feature_extractor(images)[0]
    return embeddings

def fetch_patch(slide_id, magnification_level, coords, images, start_index, end_index, num_threads=4):
    for i in range(start_index, end_index, num_threads):
        coord_x = coords[i][0]
        coord_y = coords[i][1]
        img = getTile(slide_id, magnification_level, coord_x, coord_y)
        images[i] = img
