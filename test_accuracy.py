#!/usr/bin/env python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import re
import pandas as pd
from util import test_accuracy

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("filepath", help="Sample CSV to test")
    args = parser.parse_args()

    input = pd.read_csv(args.filepath)
    accuracy = test_accuracy(input['Id'], input['Category'])
    print(accuracy)
