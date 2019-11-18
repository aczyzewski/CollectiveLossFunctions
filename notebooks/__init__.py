import os
import sys

SRC_DIR = '../src'
DATASETS_DIR = '../datasets'

# Methods
def initialize_notebook():
	""" Sets all important variables and paths in Notebook's environment """

	# Add src files into PATH
	sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), SRC_DIR))


	