import os
import numpy as np
import tifffile
import xml.etree.ElementTree as ET
from globals import *


def _read_raw_movie(path, dtype=np.uint16):
    """ Read movie from a .raw path
        First get size from the Experiment.xml file (assumed to be in the same folder)
        Then use it to retrieve the movie
    """
    raw_movie_1d = np.fromfile(path, dtype)
    tree = ET.parse(os.path.join(os.path.split(path)[0], 'Experiment.xml'))
    root = tree.getroot()

    # Find the LSM tag and get the frameRate attribute
    lsm = root.find("LSM")
    globals.SAMPLING_RATE = int(float(lsm.attrib['frameRate'].replace(',', '')))
    print('Frame rate:', SAMPLING_RATE)

    width = int(root[5].attrib['width'])
    height = int(root[5].attrib['height'])
    movie_3d = np.reshape(raw_movie_1d, (-1, height, width))
    print(f"Movie shape: {movie_3d.shape}")
    return movie_3d


def read_movie(path, dtype=np.uint16):
    """ Read a movie from path
        Each file type requires different handling
        If the GEVI used is negatively-going, flip the movie
    """

    print("Loading data...")
    movie_type = path.split('.')[-1]
    if movie_type == 'tif':
        movie = tifffile.imread(path)
    elif movie_type == 'raw':
        movie = _read_raw_movie(path, dtype=dtype)
    else:
        raise ValueError('Unknown movie type: {}'.format(movie_type))

    return movie


def basic_movie_preprocessing(movie):
    if IS_NEGATIVE_GEVI: movie = -movie

    movie = movie.astype(np.float32)

    # Subtract baseline to reduce background noise
    avg_img = np.mean(movie, axis=0)
    movie -= np.percentile(avg_img, 20)

    return movie