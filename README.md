# One-Shot Visual Imitation Learning via Meta-Learning

*A TensorFlow implementation of the paper [One-Shot Visual Imitation Learning via Meta-Learning (Finn*, Yu* et al., 2017)](https://arxiv.org/pdf/1709.04905.pdf).* Here are the instructions to run our experiments shown in the paper.

First clone the fork of the gym repo found [here](https://github.com/tianheyu927/gym), and following the instructions there to install gym. Switch to branch *mil*.

Then go to the `mil` directory and run `./scripts/get_data.sh` to download the data.

After downloading the data, training and testing scripts for MIL are available in `scripts/`.

*Note: The code only includes the simulated experiments.*



# Rachael Notes:
Installing other packages:
. venv/bin/activate
pip install numpy
pip install tensorflow
pip install imageio
pip install scipy
pip install natsort
pip install gym
pip install joblib
pip install mujoco-py==0.5.7

For mujoco, you have to download an mjkey.txt file and mjpro131 folder. You must then do:

export MUJOCO_PY_MJKEY_PATH=/Users/rachael/Documents/Spring2018Classes/semisupervised_mil/scripts/mjkey.txt

export MUJOCO_PY_MJPRO_PATH=/Users/rachael/Documents/Spring2018Classes/semisupervised_mil/scripts/mjpro131


Download mildata file and put in /scripts folder


Changes that were made to the main_replicated and data_generator files:
main_replicated.py:
	Line 18: change to specify demo_file directory './scripts/mil_data/data/sim_vision_reach/'
	Line 19: change to specify gif directory './scripts/mil_data/data/sim_vision_reach/'
	Line 20: Change flag to ‘color’
	Line 30: change to 5
	Line 21,22: change to appropriate numbers for reach

data_generator.py:
	Line 95:  Change directory to correct thing (example: '/Users/Rachael/Documents/Spring2018Classes/semisupervised_mil/scripts/mil_data/data/scale_and_bias_%s.pkl’)
	Lines 44-49: Comment out
