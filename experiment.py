import pathlib
import os
import numpy as np
import slab
import time
import tkinter
from tkinter import font as tkFont
from tkinter import ttk
from functools import partial
import freefield
from config import COLOURS, NUMBERS, COL_TO_HEX, TARGET_SEGMENT_LENGTH, SEGMENT_LENGTHS, STIM_MODEL, ELE_LOC_SPEAKERS
from utils import get_params, get_stimulus, reverse_sound, combine_sounds

SAMPLERATE = 48828

slab.Sound.set_default_samplerate(SAMPLERATE)

DIR = pathlib.Path(os.getcwd())

play_buf_multi = str(DIR) + '/data/rcx/play_buf_multi.rcx'

proc_list = [['RP2', 'RP2', play_buf_multi],
             ['RX81', 'RX8', play_buf_multi],
             ['RX82', 'RX8', play_buf_multi]]

freefield.initialize('dome', zbus=True, device=proc_list)
freefield.set_logger('WARNING')

stim_dir = pathlib.Path.resolve(DIR / "samples" / "CRM_resamp-48828")


class Experiment:
    def __init__(self, subject_id):
        self.subject_id = subject_id
        self.results_file = None
        self.trial_seq = None
        self.trial_data = dict()
        self.prev_target_params = None
        self.end_stim_timestamp = time.time()
        self.task = "single_source"
        self.plane = None
        self.level = 80
        self.is_reversed = True
        slab.set_default_level(self.level)
        self.master = None
        self.myFont = None
        self.is_screen_on = False

    def show_default_screen(self, prompt="Welcome to the Free Field"):
        if self.is_screen_on:
            self.master.destroy()
        self.master = tkinter.Tk()
        self.master.title("Welcome")
        self.master.geometry('%dx%d+%d+%d' % (1280, 720, 0, 0))
        self.myFont = tkFont.Font(root=self.master, family='Helvetica', size=20, weight=tkFont.BOLD)
        self.master.configure(background='#373F51')
        self.master.attributes("-fullscreen", True)
        button = tkinter.Button(self.master, text=prompt, highlightthickness=0, bd=0, command=self.master.destroy)
        button.pack(padx=30, pady=20)
        button["font"] = self.myFont
        self.master.mainloop()

    def initialise_UI(self, prompt="Welcome to the Free Field"):
        self.master = tkinter.Tk()
        self.master.title("Responses")
        self.master.geometry('%dx%d+%d+%d' % (1280, 720, 0, 0))
        self.myFont = tkFont.Font(root=self.master, family='Helvetica', size=36, weight=tkFont.BOLD)
        self.master.configure(background='#373F51')
        self.master.attributes("-fullscreen", True)
        self.generate_numpad()
        self.run_trial()
        self.master.mainloop()

    def run_trial(self):
        if self.trial_seq.n_remaining != 0:
            self.master.update()
            self.trial_seq.__next__()
            if self.task == "single_source":
                target_params = get_params(exclude=self.prev_target_params)
                target = get_stimulus(target_params)
                if self.is_reversed:
                    target_params["segment_length"] = self.trial_seq.this_trial
                    target, target_reverse_seed = reverse_sound(target, target_params["segment_length"])
                    target_params["reverse_seed"] = target_reverse_seed
                masker = slab.Binaural.silence(duration=target.n_samples, samplerate=target.samplerate)
                masker_params = STIM_MODEL
            elif self.task == "multi_source":
                target_params = get_params(pick={"call_sign": "Baron"}, exclude=self.prev_target_params)
                target_params["segment_length"] = TARGET_SEGMENT_LENGTH
                target = get_stimulus(target_params)
                target, target_reverse_seed = reverse_sound(target, target_params["segment_length"])
                masker_params = get_params(pick={"talker_id": target_params["talker_id"]}, exclude=target_params)
                masker = get_stimulus(masker_params)
                while abs(masker.duration - target.duration) > 0.25:
                    masker_params = get_params(pick={"talker_id": target_params["talker_id"]}, exclude=target_params)
                    masker = get_stimulus(masker_params)
                masker_params["segment_length"] = self.trial_seq.this_trial
                masker, masker_reverse_seed = reverse_sound(masker, masker_params["segment_length"])
                target_params["reverse_seed"] = target_reverse_seed
                masker_params["reverse_seed"] = masker_reverse_seed
            masker.level, target.level = self.level, self.level
            target_params["speaker_chan"] = 1
            target_params["speaker_proc"] = "RX81"
            playbuflen = max(target.n_samples, masker.n_samples)
            freefield.write(tag="playbuflen", value=playbuflen, processors="RX81")
            if self.task == "single_source":
                freefield.write(tag="data0", value=target.left.data, processors=target_params["speaker_proc"])
                freefield.write(tag="chan0", value=target_params["speaker_chan"], processors=target_params["speaker_proc"])
                freefield.write(tag="data1", value=masker.left.data, processors="RX81")
                freefield.write(tag="chan1", value=99, processors="RX81")
            elif self.task == "multi_source":
                if self.plane == "collocated":
                    masker_params["speaker_chan"] = 1
                    masker_params["speaker_proc"] = "RX81"
                    combined = combine_sounds(target, masker)
                    combined.level = self.level + 3
                    self._load_sound(0, combined, target_params)
                elif self.plane == "azimuth":
                    masker_params["speaker_chan"] = 18
                    masker_params["speaker_proc"] = "RX81"
                    self._load_sounds(target, target_params, masker, masker_params)
                elif self.plane == "elevation":
                    masker_params["speaker_chan"] = 5
                    masker_params["speaker_proc"] = "RX81"
                    self._load_sounds(target, target_params, masker, masker_params)
                elif self.plane == "front-back":
                    masker_params["speaker_chan"] = 18
                    masker_params["speaker_proc"] = "RX82"
                    masker.level += 1.78
                    self._load_sounds(target, target_params, masker, masker_params)

            print("Task", f"({self.trial_seq.this_n + 1}/{self.trial_seq.n_trials}):  \t", target_params["colour"], target_params["number"])

            freefield.write(tag='bitmask', value=8, processors='RX81')
            trial_timestamp = time.time()
            freefield.play()
            freefield.wait_to_finish_playing()
            freefield.write(tag='bitmask', value=0, processors='RX81')

            self.end_stim_timestamp = time.time()
            self._clear_buffers(target, masker)
            trial_params = {"timestamp": trial_timestamp, "index": self.trial_seq.this_n}
            self.results_file.write(trial_params, tag="trial_params")
            self.results_file.write(target_params, tag="target_params")
            self.results_file.write(masker_params, tag="masker_params")
            self.prev_target_params = target_params
        else:
            self.results_file.write(self.trial_seq, "trial_seq")
            self.master.destroy()
            end_sound = slab.Sound.read(DIR / "data" / "sounds" / "start.wav")
            freefield.write(tag="data0", value=end_sound.data, processors="RX81")
            freefield.write(tag="chan0", value=1, processors="RX81")
            freefield.play()
            print("End of task")
            # print_current_score(subject_id=self.subject_id)
            freefield.wait_to_finish_playing()
            freefield.write(tag="data0", value=np.zeros(end_sound.n_samples), processors="RX81")

    @staticmethod
    def _load_sound(tag, sound, params):
        chan = params["speaker_chan"]
        proc = params["speaker_proc"]
        freefield.write(tag=f"data{tag}", value=sound.left.data, processors=proc)
        freefield.write(tag=f"chan{tag}", value=chan, processors=proc)

    def _load_sounds(self, target, target_params, masker, masker_params):
        self._load_sound(0, target, target_params)
        self._load_sound(1, masker, masker_params)

    @staticmethod
    def _clear_buffers(target, masker):
        clearbuflen = max(target.n_samples, masker.n_samples)
        freefield.write(tag="data0", value=np.zeros(clearbuflen), processors="RX81")
        freefield.write(tag="data1", value=np.zeros(clearbuflen), processors="RX81")
        freefield.write(tag="data1", value=np.zeros(clearbuflen), processors="RX82")

    @staticmethod
    def name_to_int(name):
        return int(name[-1]) + 1

    def set_level(self, level):
        self.level = level

    def button_press(self, response_params):
        if response_params:
            print("Response:\t\t", response_params["colour"], response_params["number"])
        if self.results_file and not self.trial_seq.this_n < 0:
            response_params["timestamp"] = time.time()
            response_params["colour_correct"] = self.prev_target_params["colour"] == response_params["colour"]
            response_params["number_correct"] = self.prev_target_params["number"] == response_params["number"]
            response_params["score"] = (response_params["colour_correct"] * len(COLOURS) +
                                        response_params["number_correct"] * len(NUMBERS)) / (len(COLOURS) + len(NUMBERS))
            self.results_file.write(response_params, tag="response_params")
        self.run_trial()

    def generate_numpad(self):
        progress_bar = ttk.Progressbar(orient="horizontal", length=950)
        progress_bar.place(x=30, y=550)
        progress_step = 100 / self.trial_seq.n_trials

        def clicked(response_params):
            if self.trial_seq.this_n >= 0 and self.trial_seq.n_remaining != 0:
                progress_bar.step(progress_step)
            self.button_press(response_params)
        buttons = [[0 for x in range(len(COLOURS))] for y in range(len(NUMBERS))]
        for row, c_name in enumerate(COLOURS):
            for column, n_name in enumerate(NUMBERS):
                response_params = {"colour": c_name, "number": n_name}
                button_text = self.name_to_int(NUMBERS[n_name])
                buttons[column][row] = tkinter.Button(
                    self.master,
                    text=str(button_text),
                    bg=COL_TO_HEX[c_name],
                    highlightthickness=0,
                    bd=0,
                    activebackground="red",
                    command=partial(clicked, response_params)
                )
                buttons[column][row]['font'] = self.myFont
                buttons[column][row].grid(row=row, column=column, padx=30, pady=20)

    def run_task(self, task="multi_source", phase="experiment", direction="reversed", plane=None, n_reps=10):
        self.show_default_screen(prompt="Start")
        self.task = task
        self.plane = plane
        self.is_reversed = True if direction == "reversed" else False
        n_reps = 1 if phase == "intro" else n_reps
        self.trial_seq = slab.Trialsequence(SEGMENT_LENGTHS, n_reps=n_reps)
        self.results_file = slab.ResultsFile(subject=self.subject_id)
        exp_params = {
            "subject_id": self.subject_id,
        }
        task_params = {
            "type": task,
            "phase": phase,
            "plane": plane,
            "level": self.level
        }
        self.results_file.write(exp_params, "exp_params")
        self.results_file.write(task_params, "task_params")
        self.initialise_UI()
