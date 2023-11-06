import os
from random import Random, random
from PIL import Image
from utils.file_utils import create_dir

# =================================== Read Tiles Utils =========================================================

def getSlidePaths(path, pos_range, neg_range, random_seed):
    if pos_range[0] > pos_range[1] or neg_range[0] > neg_range[1]:
        raise Exception("[getSlidePaths]: invalid range limits")
    
    pos_slides, neg_slides = [], []
    slides = os.listdir(path)
    Random(random_seed).shuffle(slides)
    for slide_folder in slides:
        if slide_folder[4] == "0":
            neg_slides.append(slide_folder)
        else:
            pos_slides.append(slide_folder)
    
    if pos_range[1] > len(pos_slides) or neg_range[1] > len(neg_slides):
        raise Exception("[getSlidePaths]: slide upper limit out of range.\
                         Positive Slides: {}/Negative Slides: {}".format(
                        len(pos_slides, 
                        len(neg_slides))) )

    pos_slides = pos_slides[pos_range[0]:pos_range[1]] if pos_range[0] != pos_range[1] else []
    neg_slides = neg_slides[neg_range[0]:neg_range[1]] if neg_range[0] != neg_range[1] else []
    
    return pos_slides, neg_slides 



def createBagsFromFolder(path, pos_paths, neg_paths):
    bags = []
    coords = []
    labels = []

    pos_bags, pos_coords, pos_labels = createBags(path, pos_paths, 1)
   
    neg_bags, neg_coords, neg_labels = createBags(path, neg_paths, 0)

    bags = pos_bags + neg_bags
    coords = pos_coords + neg_coords
    labels = pos_labels + neg_labels
    
    return bags, coords, labels
    
    
def createBags(path, bag_list, label):
    bags = []
    coords = []
    labels = [label] * len(bag_list)
    for pos_folder in bag_list:
        bag = []
        bag_coords = []
        for tile in os.listdir("{}/{}".format(path, pos_folder)):
            img = Image.open("{}/{}/{}".format(path, pos_folder, tile))
            img = img.crop((0,0,512,512))
            tile_coords = tile.split(".")[0].split("_")[1:]
            coord = (int(tile_coords[0]), int(tile_coords[1]))
            bag.append(img)
            bag_coords.append(coord)
        
        bags.append(bag)
        coords.append(coord)
    
    return bags, coords, labels



def generateNegativeBags(folder, size=100):
    negative_bags = [bag for bag in os.listdir(folder) if bag[4] == "0"]
    print(len(negative_bags))
    for i in range(size):
        bag_size = int(random() * 150)
        new_bag_name = "bag_0_augm{id}".format(id=i)
        new_bag_path = folder + "/" + new_bag_name
        create_dir(new_bag_path)
        for i in range(bag_size):
            random_bag = negative_bags[int(random()*len(negative_bags))]
            random_bag_patches = os.listdir(folder + "/" + random_bag)
            random_patch = Image.open(folder + "/"+ random_bag + "/" + random_bag_patches[int(random()*len(random_bag_patches))])
            random_patch.save(new_bag_path+"/patch_"+str(i)+"_"+str(i)+".png")        


