# Imlo_coursework
repository for IMLO CNN coursework

To run the model run:
conda env create -f environment.yaml 
(this will create a virtual environment using conda and automatically download all the requirements from environment.yaml)

conda activate model_env
(activates the virtual environment)
or run: 
pip install -r requirements.txt
in an already activated conda environment to download the requirements

then:

python train.py
(trains the model using python)

python test.py
(runs test data)
