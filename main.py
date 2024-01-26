# TODO block button press until sound is being played
# TODO save settings as slab config files

from experiment import *
from localisation import Localisation
from analysis import plot_results

subject_id = "test_jakab"

exp = Experiment(subject_id)
loc = Localisation(subject_id)

# INTRODUCTION ==============================================================
exp.run_task(task="single_source", phase="intro", direction="forward")

# LOCALISATION TEST =========================================================
loc.run_test()

# SINGLE SENTENCE ===========================================================
exp.run_task(task="single_source", phase="experiment", direction="reversed")  # x2

# TWO SENTENCES - HORIZONTAL ================================================
exp.run_task(task="multi_source", phase="experiment", plane="azimuth")  # x2

# TWO SENTENCES - VERTICAL ==================================================
exp.run_task(task="multi_source", phase="experiment", plane="elevation")  # x2

# TWO SENTENCES - FRONT-BACK ================================================
exp.run_task(task="multi_source", phase="experiment", plane="front-back", n_reps=2)  # x2

# TWO SENTENCES - SAME LOCATION =============================================
exp.run_task(task="multi_source", phase="experiment", plane="collocated")  # x2


# RESULTS ===================================================================
plot_results(subject_id=subject_id, task_type="single_source")
plot_results(subject_id=subject_id, task_type="multi_source")
