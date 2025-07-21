import numpy as np

# TODO define per experiment

STIM_ON_PERIOD = np.arange(300, 1900)
STIM_OFF_PERIOD = np.arange(2000, 2500)
SAMPLING_RATE = 1000 # Hz

BLUE_ON = np.array([1984, 5978, 9973, 13970, 17966, 21963])
BLUE_OFF = np.array([1, 3984, 7979, 11973, 15972, 19967])
LIGHT_STIMULATION_DURATION = 1900
FRAMES_TO_SKIP = 80 # amount of frames to skip from the start
