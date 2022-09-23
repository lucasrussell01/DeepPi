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


## setup CMSSW and DeepPi0 repository

#Luca(s) add this

For this step you should be working with the `cmsenv` environment.

### Compile code:
Compile code from within the production folder:
`cd Production`
`scram b -j8`

### Running ntuples locally:
This configuration file is used to produce ntuples: `python/higgstautau_cfg_aod_inc_hps_2018.py`
To run it locally for individual files, use `cmsRun`.

### Running ntuples using crab:
To produce ntuples for all samples, jobs can be submitted using crab:
`cd python`
`python ../crab/crab_DeepPi.py --output_folder=DetectorImages_MVA --year=2018 --mc`
where the `output_folder` argument specifies where in your personal dcache space you want to files to be outputted to.


# Producing input images

Once ROOT tuples are produced, they must be processed to extract true taus, and form the `pickle` dataframes that are used as inputs for training. For maximal efficiency, this step should also be performed with the `cmsenv` environemnt active.

Switch to the `Analysis` area.:
`cd ../../Analysis`

### Get list of files:
After crab jobs have finished get the list of the files in the dcache area:
`./scripts/get_filelists.sh store/user/lrussell/DetectorImages_MVA_MC_106X_2018/ HPS_2209`
you must change the name of the directory to the dcache directory where you stored the output from the crab jobs.

Produce input images using:
`python python/GenerateImages.py --n_tau=-1 --sample=GluGluHToTauTau_M125 --split=L --save_path=Images`
Changing split option to dataset split name (A,B,C,D....). Datasets are split into groups of 50 files and each assigned a letter, this is done as there is an issue with memory leakage if more files than this are processed simultaneously.
NB: each split should take around 8 hours to run.

#TODO: make as script to run this stage as batch jobs 

#TODO: Lucas inside GenerateImages.py the paths to the filelists and the output folder are hard coded. For the time being modify these by hand "path_to_filelist". Add option for this
#TODO Lucas: remove t-notify.sh


# Training:

work in DeepPi0 directory:
`cd ..`
`source env.sh conda`

file to configure training is: Training/configs/training.yaml
update options in this file or specify them as command line arguments e.g change input file directory 

to run with all option as specified in training.yaml:

`cd Training/python`
`python TrainingNN.py experiment_name=training`

"experiment_name" is used to group models together
 

or to change and option in training.yaml:
need to specific a model_name. At the moment there are 2 options:
'DeepPi_v1' = DM classification
'DeepPi_v2' = kinematic regression

`python TrainingNN.py experiment_name=training training_cfg.Setup.n_tau=50`

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

