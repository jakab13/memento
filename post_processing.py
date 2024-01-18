import pandas as pd
from analysis import load_df

df = load_df()
df_pilot = df[df['subject_id'].str.contains("pilot")]
df_pilot = df_pilot[df_pilot["task_phase"] != "intro"]
speaker_table = pd.read_csv("speakertable_dome.csv")
stim_features = pd.read_csv("stim_features.csv")

# Insert speaker locations
df_pilot.loc[df_pilot["target_speaker_id"] == 1, ['target_speaker_azi', 'target_speaker_ele']] = 0, 0

speaker_id_locs = df_pilot["masker_speaker_id"] == 1
df_pilot.loc[speaker_id_locs, ['masker_speaker_azi', 'masker_speaker_ele']] = 0, 0

speaker_id_locs = (df_pilot["masker_speaker_id"] == 18) & (df_pilot["task_plane"] == "azimuth")
df_pilot.loc[speaker_id_locs, ['masker_speaker_azi', 'masker_speaker_ele']] = 17.5, 0

speaker_id_locs = (df_pilot["masker_speaker_id"] == 23) & (df_pilot["task_plane"] == "azimuth")
df_pilot.loc[speaker_id_locs, ['masker_speaker_azi', 'masker_speaker_ele']] = 35, 0

speaker_id_locs = (df_pilot["masker_speaker_id"] == 4) & (df_pilot["task_plane"] == "elevation")
df_pilot.loc[speaker_id_locs, ['masker_speaker_azi', 'masker_speaker_ele']] = 0, 37.5

speaker_id_locs = (df_pilot["masker_speaker_id"] == 18) & (df_pilot["task_plane"] == "elevation")
df_pilot.loc[speaker_id_locs, ['masker_speaker_azi', 'masker_speaker_ele']] = 0, 50

df_pilot["task_plane"].replace("colocated", "collocated", inplace=True)


def find_feature(filename, feature):
    feature_val = None
    if filename is not None:
        filename_orig = filename[:-17] + filename[-4:]
        features = stim_features.loc[stim_features["filename"] == filename_orig]
        feature_val = features[feature].values[0]
    return feature_val


df_pilot["target_duration"] = [find_feature(f, "duration") for f in df_pilot["target_filename"]]
df_pilot["target_centroid"] = [find_feature(f, "centroid") for f in df_pilot["target_filename"]]
df_pilot["target_flatness"] = [find_feature(f, "flatness") for f in df_pilot["target_filename"]]
df_pilot["masker_duration"] = [find_feature(f, "duration") for f in df_pilot["masker_filename"]]
df_pilot["masker_centroid"] = [find_feature(f, "centroid") for f in df_pilot["masker_filename"]]
df_pilot["masker_flatness"] = [find_feature(f, "flatness") for f in df_pilot["masker_filename"]]

df_pilot["reaction_time"] = df_pilot["response_time_diff"] - df_pilot[["target_duration", "masker_duration"]].max(axis=1)

df_pilot = df_pilot[df_pilot["reaction_time"] < 30]

df_pilot.to_csv("memento_pilot.csv")
