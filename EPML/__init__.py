import os
import sys
import getopt

from . import data_gen
from . import data_processing
from . import model_gen
from . import model_evaluation

#if __name__ == "__main__":
current_dir = os.listdir(".")
if "data" not in current_dir:
    os.makedirs("data/raw")
    os.makedirs("data/processed")
if "models" not in current_dir:
    os.makedirs("models")
if "mechanisms" not in current_dir:
    os.makedirs("mechanisms")
