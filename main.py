import tifffile
import matplotlib

from spike_analyzer import detect_spikes

matplotlib.use('TkAgg')

from roi_analyzer import roi_analysis


if __name__ == '__main__':

    path = "Z:/Adam-Lab-Shared/Data/Michal_Rubin/Dendrites/AceM-neon/AcAx3/08-10-2024-acax3-l-s2/fov4/vol/vol.tif"
    clean_path = "clean_movie.tif"
    #
    # print("Loading data...")
    # movie = tifffile.imread(path)
    # print("Cleaning the data...")
    # movie = clean_movie_pipeline(movie)
    #
    # tifffile.imwrite(clean_path, movie)

    # To skip cleanup process - load clean movie
    print("Loading clean data...")
    movie = tifffile.imread(clean_path)

    # Apply Clicky and get traces per ROI
    roi_masks, roi_traces = roi_analysis(movie)
    detect_spikes(roi_traces, movie)







