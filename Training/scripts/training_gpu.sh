#!/bin/sh
cd /vols/cms/lcr119/DeepPi
source env.sh conda
cd Training/python
python -u TrainingNN.py experiment_name=newcentre &> sumloss5e.log
# python -u TrainingNN.py experiment_name=momonly training_cfg.Setup.dropout=0.0 &> mom_5e5e.log
# python -u TrainingNN.py experiment_name=DMpred training_cfg.Setup.dropout=0.0 training_cfg.Setup.n_tau=50 training_cfg.Setup.n_epochs=5 training_cfg.Setup.model_name='DeepPi_v1' &> DM_batch50_5e5e.log