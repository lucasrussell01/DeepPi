#!/bin/sh
cd /home/hep/lcr119/DeepPi
source env.sh conda
cd Training/python
python TrainingNN.py experiment_name=optimisation
home/hep/lcr119/scripts/t-notify.sh Training Complete