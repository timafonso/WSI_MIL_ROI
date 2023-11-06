import h5py

from utils.download_tiles import createBagsFromFolder

import numpy as np
import pandas as pd
import os

import json
import requests

createBagsFromFolder("slide_data/BRCA_TP53.csv", "BRCA_TP53_20x_.hdf5", 342, "tp53", 20)
