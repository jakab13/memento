import numpy as np
import slab
import freefield
# import pyloudnorm as pyln
from utils import get_params, get_stimulus

SAMPLERATE = 48828

slab.Sound.set_default_samplerate(SAMPLERATE)

freefield.initialize("dome", default="play_birec")
freefield.set_logger('WARNING')
# meter = pyln.Meter(SAMPLERATE, block_size=0.200)

level_diffs = list()

for i in range(10):

    sound_params = get_params()
    sound = get_stimulus(sound_params)

    rec_ref = freefield.play_and_record(23, sound.left)
    level_ref = rec_ref.level.mean()
    # level_ref = meter.integrated_loudness(rec_ref.left.data)

    sound.level -= 2.4
    rec = freefield.play_and_record(4, sound.left)
    level_rec = rec.level.mean()
    # level_rec = meter.integrated_loudness(rec.left.data)

    level_diff = level_rec - level_ref
    print(level_rec, level_ref)
    level_diffs.append(level_diff)

print(np.asarray(level_diffs).mean())