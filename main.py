
from experiment import *
from analysis import plot_results

subject_id = "jakab"

exp = Experiment(subject_id)

# INTRODUCTION ==============================================================
exp.run_task(task="single_source", phase="intro", direction="forward")
# Was it loud enough?
# exp.set_level(90)  # 80dB is the default
exp.run_task(task="single_source", phase="intro", direction="reversed")

# SINGLE SENTENCE ===========================================================
exp.run_task(task="single_source", phase="experiment", direction="reversed")

# MULTIPLE SENTENCES - SAME LOCATION ========================================
exp.run_task(task="multi_source", phase="experiment", plane="colocated")

# MULTIPLE SENTENCES - HORIZONTAL ===========================================
exp.run_task(task="multi_source", phase="experiment", plane="azimuth")

# MULTIPLE SENTENCES - VERTICAL =============================================
exp.run_task(task="multi_source", phase="experiment", plane="elevation")


# RESULTS ===================================================================
plot_results(subject_id=subject_id, task_type="single_source")
plot_results(subject_id=subject_id, task_type="multi_source")

# TODO block button press until sound is being played
# TODO save settings as slab config files