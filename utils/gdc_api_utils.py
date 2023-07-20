from io import BytesIO
from PIL import Image

import json
import requests

#================================== API ENDPOINTS =============================================

FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
TILES_ENDPOINT = "https://api.gdc.cancer.gov/tile"
DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

#====================================== API PARAMS ==============================================

SAMPLE_IDS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]
LUNG_PROJECTS = ["TCGA-LUSC", "TCGA-LUAD"]

TCGA_LUSC = "TCGA-LUSC"
TCGA_LUAD = "TCGA-LUAD"

POSITIVE_IDS = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
NEGATIVE_IDS = ["10", "11", "12", "13", "14"]

file_filter = {
    "op": "and",
    "content": [
        {
            "op": "=",
            "content": {
                "field": "data_format",
                "value": "svs"
            }
        },
        {
            "op": "in",
            "content": {
                "field": "cases.project.project_id",
                "value": None
            }
        },
        {
            "op": "in",
            "content": {
                "field": "cases.samples.sample_type_id",
                "value": None
            }
        },     
    ]
}

fields = [
    "file_id",
    "cases.samples.sample_type_id",
    "cases.samples.portions.slides.percent_tumor_cells",
    "experimental_strategy"
    ]

fields = ",".join(fields)

filter_params = {
    "filters": None,
    "fields": fields,
    "format": "JSON",
    "from": "",
    "size": ""
}

#=========================================== Slide Id Files =====================================================

def getImageIDs(params):
    """Gets a list of the WSI ids and the labels associated, using the GDC API

    Args:
        params (dict): json object with the params for the slides

    Returns:
        list(string): list of the ids
        list(int): list of labels
    """

    file_filter["content"][1]["content"]["value"] = params["tcga_projs"]
    file_filter["content"][2]["content"]["value"] = params["sample_types"]

    ff = json.dumps(file_filter)

    filter_params["filters"] = ff
    filter_params["from"] = str(params["start"])
    filter_params["size"] = str(params["size"])

    slide_ids = []
    slide_labels = []
    response = requests.get(FILES_ENDPOINT, params=filter_params)
    
    json_object = json.loads(response.content)
    slide_data = json_object["data"]["hits"]

    tissue_slide_count = 0
    diag_slide_count = 0
    avg_percentage = 0
    count = 0
    percentages = []
    for slide in slide_data:
        if "portions" in slide["cases"][0]["samples"][0] and slide["cases"][0]["samples"][0]["portions"][0]["slides"][0]["percent_tumor_cells"] != None:
            avg_percentage += float(slide["cases"][0]["samples"][0]["portions"][0]["slides"][0]["percent_tumor_cells"])
            percentages.append(float(slide["cases"][0]["samples"][0]["portions"][0]["slides"][0]["percent_tumor_cells"]))
            count += 1
        if slide["experimental_strategy"] == "Tissue Slide":
            tissue_slide_count += 1
        else:
            diag_slide_count += 1
        slide_ids.append(slide["id"])
        sample_type_id = slide["cases"][0]["samples"][0]["sample_type_id"]
        label = 1 if sample_type_id[0] == "0" else 0
        slide_labels.append(label)

    print("PERC", avg_percentage/count)
    print("TISSUE", tissue_slide_count, "DIAG", diag_slide_count)
    print(percentages)

    return slide_ids, slide_labels

def getSlideData(tcga_projs, sample_types, response_fields):
    """Returns the slide metadata retrieved from the GDC API

    Args:
        tcga_projs (list<str>): list of TCGA project codes
        sample_types (list<str>): list of sample type codes
        response_fields (list<str>): list of fields to retrieve 

    Returns:
        dict: dictionary with the metadata to retrieve
    """
    file_filter["content"][1]["content"]["value"] = tcga_projs
    file_filter["content"][2]["content"]["value"] = sample_types

    ff = json.dumps(file_filter)
    r_fields = ",".join(response_fields)
    
    filter_params = {   
        "filters": ff,
        "fields": r_fields,
        "format": "JSON",
        "from": "0",
        "size": "10000"
    }

    response = requests.get(FILES_ENDPOINT, params=filter_params)
    json_response = json.loads(response.content)
    slide_data = json_response["data"]["hits"]

    return slide_data

def getTileEndpoint(slide_id, magnification_lvl, coord_x, coord_y):
    """Returns the url for the GDC API tile endpoint corresponding to a certain tile

    Args:
        slide_id (_type_): id of the WSI's tile
        magnification (int): tile's magnification level
        coord_x (int): tile's x coord
        coord_y (int): tile's y coord

    Returns:
        string: url for the endpoint
    """
    return "{tiles_endpoint}/{slide_id}?level={mag_lvl}&x={x}&y={y}".format(
        tiles_endpoint=TILES_ENDPOINT, 
        slide_id=slide_id, 
        mag_lvl=magnification_lvl, 
        x=coord_x, 
        y=coord_y)

def getTile(slide_id, magnification_lvl, coord_x, coord_y):
    """
    Returns a PIL Image object of a tile

    parameters:
    -----------
    slide_id: ID of the WSI
    amp_lvl: amplification/magnification level
    x: x coord of the tile
    y: y coord of the tile

    returns:
    --------
    PIL Image object (RGB)
    """
    with requests.Session() as session:
        slide_object = session.get(getTileEndpoint(slide_id, magnification_lvl, coord_x, coord_y), stream=True)

    return Image.open(BytesIO(slide_object.content)).convert('RGB')

