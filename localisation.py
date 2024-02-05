import slab
import freefield
import numpy as np
from utils import get_params, get_stimulus
from config import ELE_LOC_SPEAKERS


class Localisation:
    def __init__(self, subject_id):
        self.subject_id = subject_id
        self.results_file = None

    def run_test(self, n_reps=2):
        exp_params = {
            "subject_id": self.subject_id,
        }
        task_params = {
            "type": "loc_test",
            "phase": "intro",
            "plane": None,
            "level": None
        }
        self.results_file = slab.ResultsFile(subject=self.subject_id)
        self.results_file.write(exp_params, "exp_params")
        self.results_file.write(task_params, "task_params")
        trial_seq = slab.Trialsequence(ELE_LOC_SPEAKERS, n_reps=n_reps)
        for speaker_id in trial_seq:
            sound_params = get_params(pick={"call_sign": "Baron"})
            sound = get_stimulus(sound_params)
            speaker = freefield.pick_speakers(speaker_id)[0]
            chan = speaker.analog_channel
            proc = speaker.analog_proc
            freefield.write(tag="data0", value=sound.left.data, processors=proc)
            freefield.write(tag="chan0", value=chan, processors=proc)
            ele_name = None
            if speaker.elevation == 50.0:
                ele_name = "extreme top"
            elif speaker.elevation == 25.0:
                ele_name = "middle top"
            elif speaker.elevation == 0.0:
                ele_name = "centre"
            elif speaker.elevation == -25.0:
                ele_name = "middle bottom"
            elif speaker.elevation == -50.0:
                ele_name = "extreme bottom"
            print(f"Sound played from {ele_name}")
            freefield.play()
            freefield.wait_to_finish_playing()
            response = input("Where did the sound come from? Correct (0) or incorrect (1)?")
            trial_seq.add_response(response)
            freefield.write(tag="data0", value=np.zeros(sound.n_samples), processors=proc)
        self.results_file.write(trial_seq, "trial_seq")
        print("Done with localisation test")