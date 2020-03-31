## ----------------------------------------------------------------------
## > Collective Loss Functions
## A simple Makefile that helps managing the project.
## ----------------------------------------------------------------------

# | Variables
SRC_DIR = src
NOTEBOOKS_DIR = notebooks
TESTS_DIR = tests

# | Actions
help:     		## Show this help.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

test:	  		## Run tests
	@python -m unittest discover $(TESTS_DIR) -p *_test.py

notebook:		## Convert all notebooks in "notebooks" dir into Markdown.	
	@jupytext --to ipynb $(NOTEBOOKS_DIR)/*.md

markdown:		## Convert all Markdown file in "notebooks" dir into Jupyter Notebooks.
	@jupytext --to md $(NOTEBOOKS_DIR)/*.ipynb

wandb:			## Log into W&B project (the secret key has to be stored in WANDB_PROJECT_KEY env variable).
ifeq ($(WANDB_PROJECT_KEY),)
	@echo "Key not found. Please assign your project secret key to WANDB_PROJECT_KEY env variable and try again."
else
	wandb login $(WANDB_PROJECT_KEY);
endif
