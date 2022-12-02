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

Switch to the `Analysis/python` area:
`cd ../../Analysis/python`

### Get list of files:
After crab jobs have finished get the list of the files in the dcache area:

```
../scripts/get_filelists.sh store/user/dwinterb/DetectorImages_MVA_Oct06_MC_106X_2018/ HPS_centre
```

you must change the name of the directory to the dcache directory where you stored the output from the crab jobs. `HPS_centre` is an example run name used for the file list naming convention, you can pick whatever you like but will need to specify it during the next image generation step.

### Produce input images:

Image of taus are generated from ROOT files using `GenerateImagesFile.py` for each ROOT file. To complete this step quickly, jobs should be submitted for each sample type (eg ggH, VBF...) using `BatchSubmission.py`:

```
python BatchSubmission.py --sample=GluGluHToTauTau_M-125 --path_to_list=/vols/cms/lcr119/CMSSW_10_6_19/src/DeepPi/Analysis/python --d_cache=/store/user/dwinterb/DetectorImages_MVA_Oct06_MC_106X_2018/ --run_name=HPS_centre --save_path=/vols/cms/lcr119/Images/HPSCentering/
```
Here `sample` is the sample type to process, `path_to_list` is where the filelists you just generated are stored, `d_cache` is the path to the crab output in the d cache storage system, `run_name` is what you specified when generating the file list, and `save-path` is where you would like the images to be saved.

NB: Each sample will created and submit 100-1000 jobs, which take 3-7 minutes each.

### Split into Training/Validation and Evaluation

You must move the images you created into to separate folders, one for training/validation and one for evaluation. The paths to these folders will need to be specified in the training configuration file later. A Training/Validation and Evaluation split of around 85/15 is what was used previously here. 

# Training:

For neural network training you must work in the `tau-ml` environemnt, to install this, go to the `DeepPi` directory and source the environment:

```
cd ..
source env.sh conda
```

Next, navigate to the `Training/python` folder:

` cd Training/python `


### Training Configuration

The file to configure training is: `Training/configs/training.yaml`

You can either update options in this file directly or specify them as command line arguments when runnign training (e.g change input file directory).

Note that to choose which model to train (kinematic regression or DM classification) a `model_name` is specified. At the moment there are 2 options:

- 'DeepPi_v1' = DM classification

- 'DeepPi_v2' = kinematic regression

To run with all of the options as specified in `training.yaml`:

```
python TrainingNN.py experiment_name=training
```

where `experiment_name` is used to define an experiment ID, which is useful for keeping track of models/grouping several similar runs together.
 

To run with changes to options in `training.yaml`:

```
python TrainingNN.py experiment_name=training training_cfg.Setup.n_tau=50
```

where in this example the number of taus per batch was modified to 50.

### To run on interactive GPU node:

```
ssh lxgpu00
```

then source conda again and follow above instructions to run `TrainingNN.py` again.

### To run using batch GPU nodes:

Use `scripts/training_gpu.sh`.

You need to change the first `cd` command in this file by hand to match where your `DeepPi` folder is. The script will then source the environment and run all the steps for training listed above. You can modify the training command in the same manner to specify options etc.

To submit the training to the batch system:

```
qsub -q gpu.q -l h_rt=23:45:45 ../scripts/training_gpu.sh
```

### Model saving

Running training will create an `mlruns` folder within `Training/python` with an experiment ID allocated to the experiment name (e.g 0,1,2,...), this will also create a model with a run-ID (hash code) attributed to it e.g mlruns/2/f27c2b7c4c7343e7aa0db74aeec1b924 

To evaluate the model or do anything else with it (eg continue trainign), you need to specify the experiment ID and the run-ID

### To continue training an existing model

 To continue training a model (for example extending the number of epochs), modify the config file: `hydra_train.yaml`.
 
Comment the `pretrained=null` line, and uncomment the lines below, modifying the `experiemnt_id` and `run_id` options as needed:
```
# pretrained:
#   run_id : f27c2b7c4c7343e7aa0db74aeec1b924 
#   experiment_id: 2
#   starting_model: DeepPi_v1_step_final.tf
```
Models are saved after each epoch, with `DeepPi_v1` for DM, and `DeepPi_v2` for kinematic regression. In the example above the final model is used but you can also use another epoch e.g by changing `DeepPi_v1_step_final.tf` to e.g `DeepPi_v1_step_e4.tf`

### To modify loss functions:

Loss functions are defined in `Training/python/losses.py`:
- `DecayMode_loss` is loss for Decay mode classification
- `Kinematic_loss` is loss for kinematic regression  

If you modify these definitions a different loss function will be applied during training.

### Changing input variables

You can modify input variables inside `DataLoader.py`. Note that you need to make sure the information you want to use is stored in the dataframe within the pkl file produced by `GenerateImages.py`.

Inside  TrainingNN.py:
Specify the input_shape and input_types variables if you have modified the shapes or types of the inputs e.g:

```
input_shape = (((33, 33, 5), 31), 3, None)
input_types = ((tf.float32, tf.float32), tf.float32, tf.float32)
```

in this example the shapes "(33, 33, 5)" are the images, "31" is the high-level variables, 3 is the kinematic truth shape, and None is the shape of the weights (not used in this example), for types: "(tf.float32, tf.float32)" are the types of the images and the high-level variables respetivly, and "tf.float32, tf.float32" are the targets and the weights.

You must also modify the network architecture to handle the input/output shapes. The "create_vX_model" functions have input_layers - specify a shape that matches the shape of the inputs.


### Changing target definitions

If you want to change the targets that you are attemptin to predict, it is easiest to add a new generator in `DataLoader.py` e.g "get_generator_v3" - you could follow "get_generator_v2" as an example.

Inside `TrainingNN.py` there is a "run_training" function  that calls the "get_generator_vX function" - modify name here

Then apply the above instructions for modifying the shapes and types of inputs, but this time modifying the targets. You will also need to modify the output layers in the network architecture e.g "outputKin" to number of nodes you want in create_vX_model in `TrainingNN.py`. You can then modify the loss function as you like, taking into account the different outputs.


# Evaluation for DM classification

Evaluation tools can be found in the `Evaluation` directory:
```
cd Evaluation/python
```

The following scripts are provided:

- `apply_DM_training.py`: applies the DM classification model to a evaluation sample 
- `apply_Kin_training.py`: applies the kinematic models to a evaluation sample
- `kinematic_predictions.py`: plots the evaluated kinematic models e.g plots of eta, phi, and momentum distributions  
- `monitor_metrics.py`: plots the metrics e.g loss, accuracy etc..
- `confusion_matrix.py`:  plots efficiency and purity matrices for SM classification

At the moment need to modify the directory by hand inside `apply\*training.py` scripts by changing "input_dir".
Additionally, if the shapes or types of inputs and outputs have been modified also need to change these by hand.

There are two steps in evaluating models:

- Apply training (will evaluate the model on a given number of taus)
- Plot Results (plot distributions, confusion matrices etc...)

### Apply model for DM classification:

To, for example, apply the model trained above to 10k taus:
```
python apply_DM_training.py --expID=2 --runID=f27c2b7c4c7343e7aa0db74aeec1b924 --n_tau=10000
``` 



### Apply model for Kinematic regression:
```
python apply_Kin_training.py --expID=2 --runID=f27c2b7c4c7343e7aa0db74aeec1b924 --n_tau=10000
``` 
### Plot results

For now use the respective juypter notebook inside `Evaluation/notebooks`.

NB: To start a notebook session, make sure port forwarding is enabled when you connected to the cluster via ssh, specifying a port e.g:

```
ssh lcr119@lx03.hep.ph.ic.ac.uk -Y -L 1879:localhost:1879
```

Then source the environement, and run:
```
jupyter-notebook --port=1879 --no-browser
```




