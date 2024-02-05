# TODO block button press until sound is being played
# TODO save settings as slab config files

from experiment import *
from localisation import Localisation
from analysis import plot_results
from questionnaires import difficulty_assessment

subject_id = "test_jakab"

exp = Experiment(subject_id)
loc = Localisation(subject_id)

# INTRODUCTION ==============================================================
exp.run_task(task="single_source", phase="intro", direction="forward")

# LOCALISATION TEST =========================================================
loc.run_test(n_reps=1)

# SINGLE SENTENCE ===========================================================
exp.run_task(task="single_source", phase="experiment", direction="reversed", n_reps=2)  # x2

# TWO SENTENCES - HORIZONTAL ================================================
exp.run_task(task="multi_source", phase="experiment", plane="azimuth", n_reps=2)  # x2

# TWO SENTENCES - VERTICAL ==================================================
exp.run_task(task="multi_source", phase="experiment", plane="elevation", n_reps=2)  # x2

# TWO SENTENCES - FRONT-BACK ================================================
exp.run_task(task="multi_source", phase="experiment", plane="front-back", n_reps=2)  # x2

# TWO SENTENCES - SAME LOCATION =============================================
exp.run_task(task="multi_source", phase="experiment", plane="collocated", n_reps=2)  # x2

# DIFFICULTY ASSESSMENT =====================================================
difficulty_assessment(subject_id)

# RESULTS ===================================================================
plot_results(subject_id=subject_id, task_type="single_source")
plot_results(subject_id=subject_id, task_type="multi_source")
