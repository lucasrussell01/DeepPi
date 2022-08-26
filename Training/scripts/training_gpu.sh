#!/bin/sh
cd /home/hep/lcr119/BatchCode/DeepPi
source env.sh conda
cd Training/python
python -u TrainingNN.py experiment_name=training training_cfg.Setup.dropout=0.0 training_cfg.Setup.HPS_features=False &> training.log