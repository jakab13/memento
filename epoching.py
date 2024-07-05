import mne
import slab
import pandas as pd
from utils import reverse_sound

tables_folderpath = "data/tables/"
sound_folderpath = "/Users/jakabpilaszanovich/Documents/GitHub/reverse-speech/samples/CRM/original_resamp-48828/"

df = pd.read_csv(tables_folderpath + "reversed_speech.csv")

subject_id = "sub_jakab_EEG"

df_sub = df[df.subject_id == subject_id]

df_sub_single_source = df_sub[df_sub.task_type == "single_source"]
df_sub_collocated = df_sub[df_sub.task_plane == "collocated"]
df_sub_azimuth = df_sub[df_sub.task_plane == "azimuth"]

epochs_curr = list()
for block in [0, 1]:
    raw = mne.io.read_raw_brainvision(f"/Users/jakabpilaszanovich/Documents/GitHub/memento/Results/sub_jakab_EEG/multitalker_azimuth_jakab_block{block + 1}.vhdr")
    events = mne.events_from_annotations(raw)[0]
    event_id = {"30ms": 1, "60ms": 2, "90ms": 3, "120ms": 4, "150ms": 5}
    epochs_curr.append(mne.Epochs(raw, events, event_id=event_id))
    epochs_curr[block] = epochs_curr[block][:40]
    if block == 1:
        epochs = mne.concatenate_epochs([e for e in epochs_curr])

df["target_segment_length"] = df["target_segment_length"].replace({float('nan'): 0})


def generate_audio_array(row, sound_type="target"):
    target_filename = row["target_filename"]
    target_segment_length = row["target_segment_length"]
    target_seed = row["target_reverse_seed"]
    sound_filepath = sound_folderpath + target_filename
    sound = slab.Sound(sound_filepath)
    reversed_sound = reverse_sound(sound, target_segment_length)[0]
    if reversed_sound.data.shape[1] == 1:
        output = reversed_sound.data
    else:
        output = reversed_sound.data.mean(axis=1)
    return output


df["target_audio_data"] = df.apply(lambda row: generate_audio_array(row), axis=1)
df["masker_audio_data"] = df.apply(lambda row: generate_audio_array(row), axis=1)

