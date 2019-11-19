import argparse
from typing import Any

from data.downloaders.uci import UCIDatabase
from data.genreators.synthetic import SyntheticDataGenerator

def parse_args() -> Any:
    """ Reads parameters from the command line """

    parser = argparse.ArgumentParser(description='Set-up an experiment.')
    parser.add_argument('-n', '--name', type=str, help='Name of the experiment.')
    parser.add_argument('-d', '--datasets', type=str, nargs='+', help='The list of datasets that will be used in the experiment.')
    parser.add_argument('-df', '--datasetsfile', type=str, help='Path to file contains a list of datasets that will be used in the experiment.')
    parser.add_arguemnt('-c', '--config', type=str, help='Path to file that defines all parameters of the experiment.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    

