import os
import numpy as np
import tifffile
import xml.etree.ElementTree as ET

from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

import constants


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
    constants.SAMPLING_RATE = int(float(lsm.attrib['frameRate'].replace(',', '')))
    print('Frame rate:', constants.SAMPLING_RATE)

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
        movie = tifffile.imread(path).astype(np.float64)
    elif movie_type == 'raw':
        movie = _read_raw_movie(path, dtype=dtype)
    else:
        raise ValueError('Unknown movie type: {}'.format(movie_type))

    if constants.IS_NEGATIVE_GEVI: movie = -movie
    print(movie.dtype, movie.min(), movie.max())

    return movie


def save_movie_as_mp4(color_movie, filename="pca_movie.mp4", fps=10):
    n_frames = color_movie.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(color_movie[0])
    ax.axis('off')

    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, filename, dpi=100):
        for i in range(n_frames):
            im.set_data(color_movie[i])
            ax.set_title(f"{i} ms")
            writer.grab_frame()
    plt.close(fig)
    print(f"Saved video to: {filename}")