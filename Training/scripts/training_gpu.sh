#!/bin/sh
cd /home/hep/lcr119/BatchCode/DeepPi
source env.sh conda
cd Training/python
python -u TrainingNN.py experiment_name=debug training_cfg.Setup.dropout=0.0 &> compare_val_ABC.log