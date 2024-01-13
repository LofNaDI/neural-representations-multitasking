
from .digits import DigitTaskFactory
from .letters import LetterTaskFactory


def get_tasks_dict(tasks_list, root):
    assert len(tasks_list) == len(set(tasks_list))

    digit_tasks = DigitTaskFactory.list_tasks()
    letter_tasks = LetterTaskFactory.list_tasks()

    tasks_dict = {}
    for task_name in tasks_list:
        if task_name in digit_tasks:
            tasks_dict[task_name] = DigitTaskFactory.get_task(task_name)(root)
        elif task_name in letter_tasks:
            tasks_dict[task_name] = LetterTaskFactory.get_task(task_name)(root)
        else:
            raise ValueError(f'{task_name} not implemented!')

    return tasks_dict
