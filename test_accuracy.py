#!/usr/bin/env python
from pathlib import Path
import re
from argparse import ArgumentParser

project_root = Path(__file__).parent.absolute()

parser = ArgumentParser()
parser.add_argument("filepath")
args = parser.parse_args()


def collab_classifier(idx):
    return 1 if idx <= 2600 else 2 if idx <= 3375 else 3


total_count = 0
succ_count = 0
with open(project_root/args.filepath, 'r') as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        index, label = [int(w) for w in re.findall(r'\d+', line)]
        total_count += 1
        succ_count += (label == collab_classifier(index))

print(succ_count/total_count)
