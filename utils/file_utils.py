import os

#========================================= File Functions =====================================================

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        return True
    return False

def is_file(path):
    return os.path.isfile(path)

def save_img(img, path):
    if not os.path.isfile(path):
        img.save(path)
        return True
    return False

