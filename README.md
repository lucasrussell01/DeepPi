# DeepPi

End to end reconstruction of neutral pions produced in hadronic tau decays for analysis of the CP structure of the Higgs boson.This repository currently includes two training configurations: `DeepPi_v1` for decay mode classification (counting the number of pi0s), and `DeepPi_v2` for regression of kinematic properties of DM1/11 taus. This code uses low level input features, Reduced ECAL RecHits, which are available in the AOD data tier.

General guidelines for different areas:
- Production of RHTree ROOT tuples in `Production` (run with `cmsenv`)
- Tools to convert to dataframes/images in `Analysis` (best to run with `tau-ml`)
- Neural Network Training (and DataLoading) in `Training` (run with `tau-ml`)
- Neural Network Evaluation in `Evaluation` (run with `tau-ml`)

## Note on environments:

When using `cmsenv` for tuple production, it is recommended to only compile from the `Production` folder as python files in other areas may not be compatible with Python 2 and can cause compilation errors.

To install/activate `tau-ml` conda environment run `source env.sh conda` from the central DeepPi folder.

Parts of the code that you will find here were adapted from MLAnalyzer (Micheal Andrews), TauMLTools (CMS Tau POG developers) and the Imperial College Higgs->TauTau repository, many thanks to all who contributed.


-------
# ROOT Tuple production

The first step towards producing input images for the neural network is generating ROOT ntuples using the DeepPi produced in  the `Production` area.


## Setup CMSSW and DeepPi0 repository

To have access the the BDT scores previously used for DM classification by the `HIG-20-006` Analysis, you need to install a custom CMSSW environment. This code is designed to run in a modified version of `CMSSW_10_6_19`, first setup your environment:

`scramv1 project CMSSW CMSSW_10_6_19`

Navigate to the `src` directory:

`cd CMSSW_10_6_19/src/`

Enter the `cmsenv environment: 

`cmsenv`

Intialise the are for git:

`git cms-addpkg FWCore/Version`

This command will have created two remote repositories, `official-cmssw` and `my-cmssw`. It will also have created and switched to a new branch, `from-CMSSW_10_6_19`.

Add and pull the remote branch from the pre-configured `ICHiggsToTauTau` github:

```
git remote add ic-cmssw git@github.com:gputtley/cmssw.git
git pull ic-cmssw from-CMSSW_10_6_19
```

You must then compile the code:

```
scram b -j8
```

Your custom CMSSW environment is now set up! 

Finally clone this repository into the src folder:

`git clone git@github.com:lucasrussell01/DeepPi.git`

## Run ntuple production

For this step you should be working with the `cmsenv` environment.

### Compile code:
Compile code from within the production folder:

```
cd Production
scram b -j8
```

### Running ntuples locally:

This configuration file is used to produce ntuples: 

`python/higgstautau_cfg_aod_inc_hps_2018.py`

To run it locally for individual files, use `cmsRun`.

### Running ntuples using crab:

To produce ntuples for all samples, jobs can be submitted using crab:

```
cd python
python ../crab/crab_DeepPi.py --output_folder=DetectorImages_MVA --year=2018 --mc
```

where the `output_folder` argument specifies where in your personal dcache space you want to files to be outputted to.


# Producing input images

Once ROOT tuples are produced, they must be processed to extract true taus, and form the `pickle` dataframes that are used as inputs for training. For maximal efficiency, this step should also be performed with the `cmsenv` environemnt active.

Switch to the `Analysis` area:
`cd ../../Analysis`

### Get list of files:
After crab jobs have finished get the list of the files in the dcache area:
`./scripts/get_filelists.sh store/user/lrussell/DetectorImages_MVA_MC_106X_2018/ HPS_2209`
you must change the name of the directory to the dcache directory where you stored the output from the crab jobs.

###Produce input images using:

`python python/GenerateImages.py --n_tau=-1 --sample=GluGluHToTauTau_M125 --split=L --save_path=Images`

Changing split option to dataset split name (A,B,C,D....). Datasets are split into groups of 50 files and each assigned a letter, this is done as there is an issue with memory leakage if more files than this are processed simultaneously.
NB: each split should take around 8 hours to run.

#TODO: make as script to run this stage as batch jobs 

#TODO: Lucas inside GenerateImages.py the paths to the filelists and the output folder are hard coded. For the time being modify these by hand "path_to_filelist". Add option for this
#TODO Lucas: remove t-notify.sh


# Training:

For neural network training you must work in the `tau-ml` environemnt, to install this, go to the `DeepPi` directory and source the environment:

```
cd ..
source env.sh conda
```

Next, navigate to the `Training/python` folder:

` cd Training/python `


The file to configure training is: `Training/configs/training.yaml`

You can either update options in this file directly or specify them as command line arguments when runnign training (e.g change input file directory).

Note that to choose which model to train (kinematic regression or DM classification) a `model_name` is specified. At the moment there are 2 options:
'DeepPi_v1' = DM classification
'DeepPi_v2' = kinematic regression

To run with all of the options as specified in `training.yaml`:

`python TrainingNN.py experiment_name=training`

where "experiment_name" is used to define an experiment ID, which is useful for keeping track of models/grouping several similar runs together.
 

To run with changes to options in `training.yaml`:

`python TrainingNN.py experiment_name=training training_cfg.Setup.n_tau=50`

where in this example the number of taus per batch was modified to 50.

#to run on interactive GPU node:'

`ssh lxgpu00`

then source conda again and run TrainingNN.py again

#to run using batch GPU nodes use "scripts/training_gpu.sh"
need to change cd in this file by hand 

`qsub -q gpu.q -l h_rt=23:45:45 ../scripts/training_gpu.sh`

this will create an mlruns within Training/python with an experiment ID allocated to the experiment name (e.g 0,1,2,...)
this will also create a model with a run-ID (hash code) attributed to it e.g mlruns/2/f27c2b7c4c7343e7aa0db74aeec1b924 
to evaluate the model or do anytjing else with it need to specify the experiment ID and the run-ID

# to continue training an existing model
 modify the config file: hydra_train.yaml:

```
# pretrained:
#   run_id : f27c2b7c4c7343e7aa0db74aeec1b924 
#   experiment_id: 2
#   starting_model: DeepPi_v1_step_final.tf
```
uncomment those lines and modify options as needed
also use to another epoch change "DeepPi_v1_step_final.tf" to e.g "DeepPi_v1_step_e4.tf"

# to modify loss functions:
adjust losses.py
"DecayMode_loss" is loss for Decaymode classification
"Kinematic_loss" is loss for kinematic regression  

# changing input variable
can modify input varibales inside vim DataLoader.py
need to make sure information is stored in the pkl file produced by GenerateImages.py
inside TrainingNN.py:
also specify the input_shape and input_types variables if you have modified the shapes or types of the inputs 
e.g 
```
input_shape = (((33, 33, 5), 31), 3, None)
input_types = ((tf.float32, tf.float32), tf.float32, tf.float32)
```
in this example "(33, 33, 5)" are the images, "31" is the high-level variables, 3 is the kinematic truth shape, and None is the shape of the weights (not used in this example)
for types: "(tf.float32, tf.float32)" are the types of the images and the high-level variables respetivly, and "tf.float32, tf.float32" are the targets and the weights

modify the network architecture to handle the input/output shapes. The "create_vX_model" functions have input_layers - specify a shape that matched the shape of the ine inputs


# changing target definitions

need to add a new generator in DataLoader.py e.g "get_generator_v3" - can follow "get_generator_v2" as an example
then inside TrainingNN.py there is a "run_training" function  that calls the get_generator_vX function - modify name here
- see above instructions for modifying the shapes and types of targets
and modify loss function as you like
also change the output layer e.g "outputKin" to number of nodes you want in creat_vX_model in TrainingNN.py 


# evaluation for DM calssification
`cd Evaluation/python`

these scripts are provided:

apply_DM_training.py: applies the DM classification model to a evaluation sample 
apply_Kin_training.py: applies the kinematic models to a evaluation sample
kinematic_pred_distrib.py: plots the evaluated kinematic models e.g plots of eta, phi, and momentum distributions  
monitor_metrics.py: plots the metrics e.g loss, accuracy etc..
confusion_matrix.py:  plots efficiency and purity matrices for SM classification

at the moment need to modify the directory by hand inside appli\*training.py scripts by changing "input_dir"
is shapes or types of inputs and outputs have been modified also need to change these by hand by modifying "training_cfg" 

apply model for DM classification:
`python apply_DM_training.py --expID=2 --runID=f27c2b7c4c7343e7aa0db74aeec1b924 --n_tau=10000` 


apply model for Kinematic regression:
`python apply_Kin_training.py --expID=2 --runID=f27c2b7c4c7343e7aa0db74aeec1b924 --n_tau=10000` 


make plots:

for now use juypter notebook inside Evaluation/notebooks




