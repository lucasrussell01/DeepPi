#!/bin/sh
cd /home/hep/lcr119/DeepPi
source env.sh conda
cd Training/python
python -u TrainingNN.py experiment_name=etaphi training_cfg.Setup.dropout=0.0 &> training.log