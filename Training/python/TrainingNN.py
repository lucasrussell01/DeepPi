# mlflow logging inspired by DeepTau
from DataLoader import DataLoader
from setup_gpu import setup_gpu
from losses import TauLosses
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Activation, BatchNormalization, Flatten, \
                                    Concatenate, PReLU, MaxPooling2D
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger, LearningRateScheduler
import mlflow
mlflow.tensorflow.autolog(log_models=False)
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import json
import os

def conv_layer(prev_layer, filters, kernel_size=3, n=1):
    conv = Conv2D(filters, kernel_size, name="conv_{}".format(n),
                  kernel_initializer='he_uniform')(prev_layer) # kernel_regularizer=None (no reg for now)
    return conv

def pool_layer(prev_layer, n=1):
    poolsize = 3
    pool = MaxPooling2D(pool_size = poolsize, strides=poolsize, name="maxpooling_{}".format(n))(prev_layer)
    return pool

def dropout(prev_layer, n=1):
    drop = Dropout(0.2)(prev_layer) # add noise shape/dims
    return drop

def create_model(model_name):

    input_layer = Input(name="input_image", shape=(33, 33, 3))
    # four convolutional layers:
    conv1 = conv_layer(input_layer, 31, n=1)
    conv2 = conv_layer(conv1, 29, n=2)
    conv3 = conv_layer(conv2, 27, n=3) # reduce to 27x27
    # max pooling layer
    pool = pool_layer(conv3, 1) # reduce to 9x9
    # further convolutions
    conv4 = conv_layer(pool, 7, n=4)
    conv5 = conv_layer(conv4, 5, n=5)
    conv6 = conv_layer(conv5, 3, n=6)
    conv7 = conv_layer(conv6, 1, n=7) # reduce to 1x1
    # flatten output
    flat = Flatten(name="flatten")(conv7) # reshape to 3 
    # dense layers
    dense1 = Dense(3, name="dense1", kernel_initializer='he_uniform')(flat)
    dense2 = Dense(3, name="dense2", kernel_initializer='he_uniform')(dense1)
    # softmax output
    output = Activation("softmax", name="output")(dense2)

    # create model
    model = Model(input_layer, output, name=model_name)

    return model

def compile_model(model):

    opt = tf.keras.optimizers.Nadam(learning_rate=1e-4)
    accuracy = tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)
    model.compile(loss=TauLosses.xentropyloss, optimizer=opt, metrics=accuracy)
    # mlflow log
    mlflow.log_dict("accuracy", 'input_cfg/metric_names.json')

def run_training(model, data_loader):

    # load generators
    gen_train = data_loader.get_generator(primary_set = True)
    gen_val = data_loader.get_generator(primary_set = False)

    # datasets from generators
    input_shape = ((33, 33, 3), None)
    input_types = (tf.float32, tf.float32)
    data_train = tf.data.Dataset.from_generator(
        gen_train, output_types = input_types, output_shapes = input_shape
        ).prefetch(tf.data.AUTOTUNE).batch(data_loader.n_tau)
    data_val = tf.data.Dataset.from_generator(
        gen_val, output_types = input_types, output_shapes = input_shape
        ).prefetch(tf.data.AUTOTUNE).batch(data_loader.n_tau)

    # logs/callbacks
    model_name = data_loader.model_name
    log_name = f"{model_name}_step"
    csv_log_file = "metrics.log"
    if os.path.isfile(csv_log_file):
        close_file(csv_log_file)
        os.remove(csv_log_file)
    csv_log = CSVLogger(csv_log_file, append=True)
    # time_checkpoint = TimeCheckpoint(12*60*60, log_name)
    callbacks = [csv_log] # [time_checkpoint, csv_log]

    # Run training
    fit = model.fit(data_train, validation_data = data_val, epochs = data_loader.n_epochs, 
                    initial_epoch = data_loader.epoch, callbacks = callbacks)
    model_path = f"{log_name}_final.tf"
    model.save(model_path, save_format="tf")

    # mlflow logs
    for checkpoint_dir in glob(f'{log_name}*.tf'):
         mlflow.log_artifacts(checkpoint_dir, f"model_checkpoints/{checkpoint_dir}")
    mlflow.log_artifacts(model_path, "model")
    mlflow.log_artifact(csv_log_file)
    mlflow.log_param('model_name', model_name)

    return fit

@hydra.main(config_path='.', config_name='hydra_train')
def main(cfg: DictConfig) -> None:
    # set up mlflow experiment id
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
    if experiment is not None:
        run_kwargs = {'experiment_id': experiment.experiment_id}
    else: # create new experiment
        experiment_id = mlflow.create_experiment(cfg.experiment_name)
        run_kwargs = {'experiment_id': experiment_id}
        

    # run the training with mlflow tracking
    with mlflow.start_run(**run_kwargs) as main_run:
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id
        
        # load configs
        setup_gpu(cfg.gpu_cfg)
        training_cfg = OmegaConf.to_object(cfg.training_cfg) # convert to python dictionary
        dataloader = DataLoader(training_cfg)

        # main training
        model = create_model(dataloader.model_name)
        compile_model(model)
        fit = run_training(model, dataloader)

        # log NN params
        for net_type in ['tau_net', 'comp_net', 'comp_merge_net', 'conv_2d_net', 'dense_net']:
            mlflow.log_params({f'{net_type}_{k}': v for k,v in cfg.training_cfg.SetupNN[net_type].items()})
        with open(to_absolute_path(f'{cfg.path_to_mlflow}/{run_kwargs["experiment_id"]}/{run_id}/artifacts/model_summary.txt')) as f:
            for l in f:
                if (s:='Trainable params: ') in l:
                    mlflow.log_param('n_train_params', int(l.split(s)[-1].replace(',', '')))

        # log training related files
        mlflow.log_dict(training_cfg, 'input_cfg/training_cfg.yaml')
        mlflow.log_artifact(scaling_cfg, 'input_cfg')
        mlflow.log_artifact(to_absolute_path("Training_NN.py"), 'input_cfg')

        # log hydra files
        mlflow.log_artifacts('.hydra', 'input_cfg/hydra')
        mlflow.log_artifact('Training_NN.log', 'input_cfg/hydra')

        # log misc. info
        mlflow.log_param('run_id', run_id)
        mlflow.log_param('git_commit', _get_git_commit(to_absolute_path('.')))
        print(f'\nTraining has finished! Corresponding MLflow experiment name (ID): {cfg.experiment_name}({run_kwargs["experiment_id"]}), and run ID: {run_id}\n')
        mlflow.end_run()

        # Temporary workaround to kill additional subprocesses that have not exited correctly
        # try:
        #     current_process = psutil.Process()
        #     children = current_process.children(recursive=True)
        #     for child in children:
        #         child.kill()
        # except:
        #     pass

if __name__ == '__main__':
    main()