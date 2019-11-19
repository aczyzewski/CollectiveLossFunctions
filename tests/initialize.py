import os
import sys

SRC_DIR = '../src'
DATASETS_DIR = '../datasets'

# Methods
def initialize_context():
	""" Sets all important variables and paths in Notebook's environment """

	# Add src files into PATH
	sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), SRC_DIR))