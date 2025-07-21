import tifffile
import matplotlib.pyplot as plt

from clean_pipeline import clean_movie_pipeline
from experiment_constants import *


def get_average_image(movie):
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


if __name__ == '__main__':
    # TODO fill path
    movie = tifffile.imread("")
    movie = clean_movie_pipeline(movie)

