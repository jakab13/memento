import slab


def difficulty_assessment(subject_id):
    results_file = slab.ResultsFile(subject=subject_id)
    exp_params = {"subject_id": subject_id}
    task_params = {"type": "questionnaire", "phase": "difficulty"}
    results_file.write(exp_params, "exp_params")
    results_file.write(task_params, "task_params")
    tasks = ["azimuth", "elevation", "front-back", "collocated"]
    for task in tasks:
        response = input(f"Difficulty assesment for {task} (1-5)")
        QnA_item = {"task": task,
                    "difficulty": response}
        results_file.write(QnA_item, "assessment")

