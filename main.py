import tifffile
import matplotlib

from SNAPT import main_analysis

matplotlib.use('TkAgg')

from clean_pipeline import clean_movie_pipeline
from spike_analyzer import extract_spike_triggered_average, play_movie, extract_peri_spike_movies, click_and_plot_traces

from roi_analyzer import roi_analysis


if __name__ == '__main__':
    path = "Z:/Adam-Lab-Shared/Data/Michal_Rubin/Dendrites/AceM-neon/AcAx3/28-10-2024-acax3-s4-awake/fov1/vol_001/vol.tif"
    clean_path = "clean_movie.tif"
    #
    # print("Loading data...")
    # movie = tifffile.imread(path)
    # movie = -movie  # Flip polarity for negatively going GEVI
    #
    # print("Cleaning the data...")
    # movie = clean_movie_pipeline(movie)
    #
    # tifffile.imwrite(clean_path, movie)

    # To skip cleanup process - load clean movie
    print("Loading clean data...")
    movie = tifffile.imread(clean_path)

    # Apply Clicky and get traces per ROI
    roi_masks, roi_traces = roi_analysis(movie)
    spike_frames = extract_spike_triggered_average(roi_traces)
    avg_spike_movie = extract_peri_spike_movies(spike_frames, movie, roi_traces)
    # click_and_plot_traces(avg_spike_movie)
    main_analysis(avg_spike_movie, movie)
    input("Press Enter to exit...")






