# TODO block button press until sound is being played
# TODO save settings as slab config files

from experiment import *
from analysis import plot_results

subject_id = "pilot_lukas"

exp = Experiment(subject_id)

# INTRODUCTION ==============================================================
exp.run_task(task="single_source", phase="intro", direction="forward")

# SINGLE SENTENCE ===========================================================
exp.run_task(task="single_source", phase="experiment", direction="reversed")  # x2

# MULTIPLE SENTENCES - HORIZONTAL ===========================================
exp.run_task(task="multi_source", phase="experiment", plane="azimuth")  # x2

# MULTIPLE SENTENCES - VERTICAL =============================================
exp.run_task(task="multi_source", phase="experiment", plane="elevation")  # x2

# MULTIPLE SENTENCES - SAME LOCATION ========================================
exp.run_task(task="multi_source", phase="experiment", plane="colocated")  # x2


# RESULTS ===================================================================
plot_results(subject_id=subject_id, task_type="single_source")
plot_results(subject_id=subject_id, task_type="multi_source")
