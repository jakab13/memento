import pandas as pd
from analysis import load_df

df = load_df()
df_pilot = df[df['subject_id'].str.contains("pilot")]
df_pilot = df_pilot[df_pilot["task_phase"] != "intro"]

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

df_pilot.to_csv("memento_pilot.csv")
