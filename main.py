import os
import numpy as np
import tifffile
import xml.etree.ElementTree as ET
import matplotlib

matplotlib.use('TkAgg')

from globals import *
from clean_pipeline import clean_movie_pipeline
from roi_analyzer import roi_analysis
from spike_analyzer import extract_roi_sta, build_spike_triggered_movie, get_bright_pixel_mask
from SNAPT import snapt_pipeline, get_movie_pca


def read_raw_movie(path, dtype=np.uint16):
    raw_movie_1d = np.fromfile(path, dtype)
    tree = ET.parse(os.path.join(os.path.split(path)[0], 'Experiment.xml'))
    root = tree.getroot()

    # Find the LSM tag and get the frameRate attribute
    lsm = root.find("LSM")
    SAMPLING_RATE = int(float(lsm.attrib['frameRate'].replace(',', '')))
    print('Frame rate:', SAMPLING_RATE)

    width = int(root[5].attrib['width'])
    height = int(root[5].attrib['height'])
    movie_3d = np.reshape(raw_movie_1d, (-1, height, width))
    print(f"Movie shape: {movie_3d.shape}")
    return movie_3d


if __name__ == '__main__':
    path = "Z:/Adam-Lab-Shared/Data/Michal_Rubin/Dendrites/AceM-neon/AcAx3/28-10-2024-acax3-s4-awake/fov1/vol_001/vol.tif"
    # path = "Z:/Adam-Lab-Shared/Data/Efrat_Sheinbach/Imaging_Data/2023-07-09_088R1/R-FOV1/spont_1000hz_15sec/Image_001_001.raw"
    clean_path = "clean_movie.tif"

    print("Loading data...")
    movie = tifffile.imread(path)
    # movie = read_raw_movie(path)

    if IS_NEGATIVE_GEVI:
        movie = -movie  # Flip polarity for negatively going GEVI

    print("Cleaning the data...")
    movie = clean_movie_pipeline(movie)
    tifffile.imwrite(clean_path, movie)

    # To skip cleanup process - load clean movie
    # print("Loading clean data...")
    # movie = tifffile.imread(clean_path)

    # Apply Clicky and get traces per ROI
    roi_masks, roi_vertices, roi_traces = roi_analysis(movie)

    # Analyze first ROI (soma)
    first_roi_trace = roi_traces[0]
    window_size_ms = 50
    window_size = int(window_size_ms * SAMPLING_RATE / 1000)
    sta, spike_indices = extract_roi_sta(first_roi_trace, window_size)

    bright_pixel_mask = get_bright_pixel_mask(movie)
    sta_movie = build_spike_triggered_movie(spike_indices, movie, window_size, mask=bright_pixel_mask)

    snapt_pipeline(sta_movie, roi_masks[0])

    # Try SNAPT again on the PCA movie
    pca_movie = get_movie_pca(sta_movie)
    pca_movie[:, ~bright_pixel_mask] = 0
    snapt_pipeline(pca_movie, roi_masks[0])

    input("Press Enter to exit...")






