# DeepPi
End to end reconstruction of neutral pions produced in hadronic tau decays for analysis of the CP structure of the Higgs boson. 

Draft README document. 
- Production of RHTree ROOT tuples in `Production` (run with `cmsenv`)
- Tools to convert to dataframes/images in `Analysis` (best to run with `tau-ml`)
- NN Training and DataLoading in `Training` (run with `tau-ml`)
- NN Evaluation in `Evaluation` (run with `tau-ml`)

When using `cmsenv` for tuple production, it is recommended to only compile from the `Production` folder as python files in other areas may not be compatible with Python 2 and can cause compilation errors.

To install/activate `tau-ml` conda environment run `source env.sh conda`.

Parts of the code in this repository were originally written by DeepTau developpers and Micheal Andrews.


-------

# running ntuples

# setup CMSSW and DeepPi0 repository
#....
#Luca(s) add this

#compile code:
compile code from within the production folder:
`cd Production`
`scram b -j8`

# running ntuples locally
this config is used to produce ntuples: python/higgstautau_cfg_aod_inc_hps_2018.py
run it using cmsRun

# produce ntuples for all samples using crab:
`cd python`
`python ../crab/crab_DeepPi.py --output_folder=DetectorImages_MVA --year=2018 --mc`

# producing input images
`cd ../../Analysis`

# after crab jobs have finished get the list of the files in the dcache area:
`./scripts/get_filelists.sh store/user/lrussell/DetectorImages_MVA_MC_106X_2018/ HPS_2209`

change the name of the directory to the dcache directory where you stored the output from the crab jobs

then produce images using
`python python/GenerateImages.py --n_tau=-1 --sample=GluGluHToTauTau_M125 --split=L --save_path=Images`
changing split option to dataset split name (A,B,C,D....)
each split takes ~8 hours tpo run
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

