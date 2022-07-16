import os
import shutil
import random
import math
from argparse import ArgumentParser
from utils import get_filenames

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--percentage", type=int, default=10)
    args = parser.parse_args()

    save_dir = 'data_' + str(args.percentage)
    os.mkdir(save_dir)
    task_list = ['summarize', 'translate', 'refine', 'concode', 'defect']
    for task in task_list:
        task_data_dir = os.path.join(save_dir, task)
        os.mkdir(task_data_dir)
        if task == 'summarize':
            sub_tasks = ['ruby', 'javascript', 'go', 'python', 'java', 'php']
        elif task == 'translate':
            sub_tasks = ['java-cs', 'cs-java']
        elif task == 'refine':
            sub_tasks = ['small', 'medium']
        else:
            sub_tasks = ['none']
        for sub_task in sub_tasks:
            if sub_task != 'none':
                sub_task_data_dir = os.path.join(task_data_dir, sub_task)
                os.mkdir(sub_task_data_dir)
            else:
                sub_task_data_dir = task_data_dir

            train_fn, dev_fn, test_fn = get_filenames('data', task, sub_task)
            shutil.copy(dev_fn, sub_task_data_dir)
            shutil.copy(test_fn, sub_task_data_dir)
            new_train_fn = train_fn.replace('data/', save_dir + '/')
            with open(train_fn, 'r') as f:
                lines = f.readline()
                lines = [line.strip() for line in lines]

            random.seed(1234)
            lines = random.sample(lines, math.ceil((args.percentage / 100) * len(lines)))

            with open(new_train_fn, 'w') as f:
                for line in lines:
                    f.write(line)
                    f.write('\n')


