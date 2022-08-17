#!/bin/sh
cd /home/hep/lcr119/BatchCode/DeepPi
source env.sh conda
cd Training/python
python -u TrainingNN.py experiment_name=gpurun training_cfg.Setup.dropout=0.2 &> train_out.log