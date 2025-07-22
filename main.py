import tifffile
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from clean_pipeline import clean_movie_pipeline
from experiment_constants import *
from clean_pipeline import compute_intensity, regress_out_poly2


def get_average_image(movie, stimulus_data_available=False):
    """ Present a 2D image of mean value for each pixel across time """
    fig, ax = plt.subplots(1, 2, figsize=(20, 4))

    average_image = np.mean(movie, axis=0)

    im_original = ax[0].imshow(average_image, cmap='gray')
    fig.colorbar(im_original, ax=ax[0])
    ax[0].set_title('Average image')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    # Baseline correction - suppress low intensity background
    baseline_image = np.percentile(average_image, 20)
    average_image -= baseline_image

    im_corrected = ax[1].imshow(average_image, cmap='gray')
    fig.colorbar(im_corrected, ax=ax[1])
    ax[1].set_title('Average image With Baseline Correction')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')

    plt.show()

    if stimulus_data_available:
        # Get average images during interesting periods
        avg_img_stim_on = np.mean(movie[STIM_ON_PERIOD, :, :], axis=0)
        avg_img_stim_off = np.mean(movie[STIM_OFF_PERIOD, :, :], axis=0)

        # Baseline correction
        avg_img_stim_on -= np.percentile(avg_img_stim_on, 20)
        avg_img_stim_off -= np.percentile(avg_img_stim_off, 20)

        fig, ax = plt.subplots(1, 2, figsize=(20, 4))

        im_stim_on = ax[0].imshow(avg_img_stim_on, cmap='gray')
        fig.colorbar(im_stim_on, ax=ax[0])
        ax[0].set_title('Average Image During Stimulus ON')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')

        im_stim_off = ax[1].imshow(avg_img_stim_off, cmap='gray')
        fig.colorbar(im_stim_off, ax=ax[1])
        ax[1].set_title('Average Image During Stimulus OFF')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')

        plt.show()


def get_average_images_during_stimulus(movie):
    assert len(BLUE_ON) == len(BLUE_OFF), "Amount of stimulus start and end time points should be the same"

    blocks_num = len(BLUE_ON)
    num_epochs = len(BLUE_OFF)
    blue_on_periods = np.array([np.arange(start + FRAMES_TO_SKIP, start + LIGHT_STIMULATION_DURATION) for start in BLUE_ON])
    blue_off_periods = np.array([np.arange(start + FRAMES_TO_SKIP, start + LIGHT_STIMULATION_DURATION) for start in BLUE_OFF])

    intensity = compute_intensity(movie)
    t = np.arange(len(intensity))

    plt.figure(figsize=(10, 10))
    plt.plot(t, intensity, 'k-')
    for i in range(num_epochs):
        plt.plot(blue_on_periods[i], intensity[blue_on_periods[i]], 'b-')
        plt.plot(blue_off_periods[i], intensity[blue_off_periods[i]], 'r-')
    plt.title("Intensity with Blue On/Off Periods")
    plt.show()

    # Build stimulus-aligned movie blocks
    nframes, nrow, ncol = movie.shape
    movie_on_blocks = np.zeros((blocks_num, BLOCK_LENGTH, nrow, ncol))
    movie_off_blocks = np.zeros_like(movie_on_blocks)

    for i in range(blocks_num):
        movie_on = movie[blue_on_periods[i]]
        movie_off = movie[blue_off_periods[i]]

        # TODO understand why in original code skipped 50 and not 80
        movie_on_blocks[i] = movie_on[FRAMES_TO_SKIP:BLOCK_LENGTH + FRAMES_TO_SKIP]
        movie_off_blocks[i] = movie_off[FRAMES_TO_SKIP:BLOCK_LENGTH + FRAMES_TO_SKIP]

        # TODO understand this
        # align epochs across time (for consistency)
        if i > 0:
            movie_on_blocks[i] -= np.mean(movie_on_blocks[i, :100], axis=0)
            movie_off_blocks[i] -= np.mean(movie_off_blocks[i, :100], axis=0)

    # TODO understand this
    movie_on_blocks = movie_on_blocks.transpose(0, 2, 3, 1).reshape((-1, nrow, ncol))
    movie_off_blocks = movie_off_blocks.transpose(0, 2, 3, 1).reshape((-1, nrow, ncol))

    movie_on_blocks = regress_out_poly2(movie_on_blocks)
    movie_off_blocks = regress_out_poly2(movie_off_blocks)

    plt.figure()
    plt.plot(np.mean(movie_on_blocks, axis=(1, 2)), label="Blue On")
    plt.plot(np.mean(movie_off_blocks, axis=(1, 2)), label="Blue Off")
    plt.legend()
    plt.title("Intensity After Drift Removal")


if __name__ == '__main__':

    path = "Z:/Adam-Lab-Shared/Data/Michal_Rubin/Dendrites/AceM-neon/AcAx3/08-10-2024-acax3-l-s2/fov4/vol/vol.tif"
    clean_path = "clean_movie.tif"
    print("Loading data...")
    movie = tifffile.imread(path)
    print("Cleaning the data...")
    movie = clean_movie_pipeline(movie)
    tifffile.imwrite(clean_path, movie)
    # Observe average images
    get_average_image(movie)



