import slab
import random
import os
import pathlib

from config import TALKERS, CALL_SIGNS, COLOURS, NUMBERS, COL_TO_HEX, STIM_MODEL

DIR = pathlib.Path(os.getcwd())
stim_dir = pathlib.Path.resolve(DIR / "samples" / "CRM_resamp-48828")

slab.Sound.set_default_samplerate(48828)


def reverse_sound(sound, segment_length, overlap=0.005):
    if segment_length == 0 or segment_length is None:
        return sound, None
    segment_length = slab.Signal.in_samples(segment_length, sound.samplerate)
    overlap = int(overlap * sound.samplerate)
    reversed_sound = slab.Binaural.silence(duration=sound.n_samples, samplerate=sound.samplerate)
    idx = random.randrange(segment_length) - segment_length
    idx_seed = (idx + segment_length) / sound.samplerate
    while idx < sound.n_samples:
        start = max(idx - overlap, 0)
        end = min(idx + segment_length + overlap, sound.n_samples)
        snippet = sound.data[start:end]
        reversed_snippet = slab.Binaural(snippet[::-1], samplerate=sound.samplerate)
        if reversed_snippet.n_samples > overlap * 2:
            reversed_snippet = reversed_snippet.ramp(duration=overlap)
        reversed_sound.data[start:end] += reversed_snippet
        idx += segment_length
    reversed_sound = reversed_sound.ramp(duration=overlap, when="offset")
    reversed_sound.left = reversed_sound.left.filter(frequency=16000, kind="lp")
    reversed_sound.right = reversed_sound.right.filter(frequency=16000, kind="lp")
    return reversed_sound, idx_seed


def get_params(pick=None, exclude=None):
    pick = pick or dict()
    exclude = exclude or dict()
    out_params = STIM_MODEL.copy()
    out_params["talker_id"] = pick.get("talker_id") or random.choice(list(TALKERS))
    out_params["call_sign"] = pick.get("call_sign") or random.choice([cs for cs in CALL_SIGNS if cs != exclude.get("call_sign")])
    out_params["colour"] = random.choice([c for c in COLOURS if c != exclude.get("colour")])
    out_params["number"] = random.choice([n for n in NUMBERS if n != exclude.get("number")])
    out_params["filename"] = params_to_filename(out_params)
    return out_params


def get_stimulus(params):
    if params is not None:
        path = stim_dir / params["filename"]
        stimulus = slab.Binaural(path)
    else:
        stimulus = slab.Binaural.silence(duration=1)
    return stimulus


def params_to_filename(params):
    talker_id = params["talker_id"]
    call_sign_id = name_to_id(params["call_sign"], "call_sign")
    colour_id = name_to_id(params["colour"], "colour")
    number_id = name_to_id(params["number"], "number")
    file_name = talker_id + call_sign_id + colour_id + number_id + "_resamp-48828.wav"
    return file_name


def combine_sounds(target, masker):
    max_duration = max([masker.n_samples, target.n_samples])
    combined = slab.Binaural.silence(duration=max_duration, samplerate=target.samplerate)
    combined.data[:target.n_samples] = target.data
    combined.data[:masker.n_samples] += masker.data
    combined.data /= 2
    return combined


def name_to_id(name, param_type):
    if param_type == "call_sign":
        param_id = CALL_SIGNS[name]
    elif param_type == "colour":
        param_id = COLOURS[name]
    elif param_type == "number":
        param_id = NUMBERS[name]
    return param_id



