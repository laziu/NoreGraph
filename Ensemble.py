#!/usr/bin/env python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', dest='in_files', nargs='+', help='Multiple CSV to input.')
    parser.add_argument('-o', dest='out_file', default='runs/test_sample_ensembled.csv', help='Output CSV path.')
    args = parser.parse_args()
    print(args)

    inputs = [pd.read_csv(f) for f in args.in_files]
    result = pd.DataFrame(columns=["Id", "Category"])
    result.iloc[:, 0] = inputs[0].iloc[:, 0]

    for i in range(len(inputs[0].iloc[:, 1])):
        votes = {}
        for input in inputs:
            c = input.iloc[i, 1]
            votes[c] = (votes[c] if c in votes else 0) + 1
        result.iloc[i, 1] = max(votes, key=votes.get)

    result.to_csv(args.out_file, index=False)
