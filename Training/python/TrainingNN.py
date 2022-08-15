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
from glob import glob

def layer_ending(layer, n, dim2d = True, dropout=0): #, activation, dropout_rate # add these from cfg later
    norm_layer = BatchNormalization(name="norm_{}".format(n))(layer)
    if dim2d: # if conv or pooling
        activation_layer = PReLU(shared_axes=[1, 2], name='activation_{}'.format(n))(norm_layer) # share image dims
    else:
        activation_layer = PReLU(name='activation_{}'.format(n))(norm_layer)
    if dropout!=0:
        final = Dropout(dropout, name="dropout_{}".format(n))(activation_layer)
        return final
    else: 
        return activation_layer 
   

def conv_block(prev_layer, channels, kernel_size=3, n=1, dropout=0):
    conv = Conv2D(channels, kernel_size, name="conv_{}".format(n),
                  activation='relu', kernel_initializer='he_uniform')(prev_layer) # kernel_regularizer=None (no reg for now)
    out = layer_ending(conv, n, dropout=dropout)
    return out

def pool_block(prev_layer, n=1):
    poolsize = 3
    pool = MaxPooling2D(pool_size = poolsize, strides=poolsize, name="maxpooling_{}".format(n))(prev_layer)
    out = layer_ending(pool, n)
    return out

def dense_block(prev_layer, size, n=1):
    dense = Dense(size, name="dense_{}".format(n), kernel_initializer='he_uniform')(prev_layer)
    out = layer_ending(dense, n, dim2d=False)
    return out

def create_model(model_name):



    # # No pooling model:
    channels = 9
    input_layer = Input(name="input_image", shape=(33, 33, 5))
    # # convolutional layers:
    conv0 = conv_block(input_layer, 24, kernel_size=1, n=0) #31
    conv1 = conv_block(conv0, 15, kernel_size=3, n=1) #31
    conv2 = conv_block(conv1, channels, n=2) #29 
    conv3 = conv_block(conv2, channels, n=3) #27
    conv4 = conv_block(conv3, channels, n=4) #25
    conv5 = conv_block(conv4, channels, n=5) #23
    conv6 = conv_block(conv5, channels, n=6) #21
    conv7 = conv_block(conv6, channels, n=7) #19
    conv8 = conv_block(conv7, channels, n=8) #17
    conv9 = conv_block(conv8, channels, n=9) #15
    conv10 = conv_block(conv9, channels, n=10) #13
    conv11 = conv_block(conv10, channels, n=11) #11
    conv12 = conv_block(conv11, channels, n=12) #9
    conv13 = conv_block(conv12, channels, n=13) #7
    conv14 = conv_block(conv13, channels, n=14) #5

    flat = Flatten(name="flatten")(conv14) # 75 
    flat_size = channels * 25
    dense1 = dense_block(flat, flat_size, n=15)
    dense2 = dense_block(dense1, flat_size, n=16)
    dense3 = dense_block(dense2, flat_size, n=17)
    dense4 = dense_block(dense3, 100, n=18)
    dense5 = dense_block(dense4, 3, n=19)
    # softmax output
    output = Activation("softmax", name="output")(dense5)

    # new archi with pooling: 

        # try new:
    # conv0 = conv_block(input_layer, 24, dropout=0.2, kernel_size=1, n=0) #31
    # conv1 = conv_block(conv0, 24, dropout=0.2, kernel_size=1, n=1) #31
    # conv2 = conv_block(conv1, 16, kernel_size=5,  dropout=0.2, n=2) #31
    # conv3 = conv_block(conv2, 16, kernel_size=1,  dropout=0.2, n=3) # 27
    # conv4 = conv_block(conv3, 16, kernel_size=1, dropout=0.2, n=4) # 27
    # conv5 = conv_block(conv4, 12,  kernel_size=5,  dropout=0.2, n=5) # 27
    # conv6 = conv_block(conv5, 12,  kernel_size=1,  dropout=0.2, n=6) # 23
    # conv7 = conv_block(conv6, 12,  kernel_size=3,  dropout=0.2, n=7) # 21
    # conv8 = conv_block(conv7, 12,  kernel_size=3,  dropout=0.2, n=8) # 19
    # conv9 = conv_block(conv8, 12,  kernel_size=3,  dropout=0.2, n=9) # 17
    # conv10 = conv_block(conv9, 12,  kernel_size=3,  dropout=0.2, n=10) # 15
    # conv11 = conv_block(conv10, 9,  dropout=0.2, n=11) #11
    # conv12 = conv_block(conv11, 9,  dropout=0.2, n=12) #9
    # conv13 = conv_block(conv12, 9,  dropout=0.2, n=13) #7
    # conv14 = conv_block(conv13, 9,  dropout=0.2, n=14) #5
    # flat = Flatten(name="flatten")(conv14) # 75 
    # flat_size = 25 * 9
    # dense1 = dense_block(flat, flat_size, n=15)
    # dense2 = dense_block(dense1, flat_size, n=16)
    # dense3 = dense_block(dense2, flat_size, n=17)
    # dense4 = dense_block(dense3, 100, n=18)
    # dense5 = dense_block(dense4, 3, n=19)

    # old non overtrained:
    # convolutional layers:
    # conv1 = conv_block(input_layer, channels, n=1) #31
    # conv2 = conv_block(conv1, channels, n=2) #29 
    # conv3 = conv_block(conv2, channels, n=3) #27
    # conv4 = conv_block(conv3, channels, n=4) #25
    # conv5 = conv_block(conv4, channels, n=5) #23
    # conv6 = conv_block(conv5, channels, n=6) #21
    # conv7 = conv_block(conv6, channels, n=7) #19
    # conv8 = conv_block(conv7, channels, n=8) #17
    # conv9 = conv_block(conv8, channels, n=9) #15
    # conv10 = conv_block(conv9, channels, n=10) #13
    # conv11 = conv_block(conv10, channels, n=11) #11
    # conv12 = conv_block(conv11, channels, n=12) #9
    # conv13 = conv_block(conv12, channels, n=13) #7
    # conv14 = conv_block(conv13, channels, n=14) #5

    # flat = Flatten(name="flatten")(conv14) # 75 
    # dense1 = dense_block(flat, 75, n=15)
    # dense2 = dense_block(dense1, 75, n=16)
    # dense3 = dense_block(dense2, 75, n=17)
    # dense4 = dense_block(dense3, 75, n=18)
    # dense5 = dense_block(dense4, 3, n=19)

    # output = Activation("softmax", name="output")(dense5)
    

    # create model
    model = Model(input_layer, output, name=model_name)

    return model

def compile_model(model):

    opt = tf.keras.optimizers.Nadam(learning_rate=1e-4)
    accuracy = tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)

    strmetrics = ["TauLosses.xentropyloss"]#["accuracy", "TauLosses.xentropyloss"]
    metrics = [accuracy]
    for m in strmetrics:
        if "TauLosses" in m:
            m = eval(m)
        metrics.append(m)

    model.compile(loss=TauLosses.xentropyloss, optimizer=opt, metrics=metrics)
    # mlflow log
    metrics = {'categorical_accuracy': '', 'xentropyloss': ''}
    mlflow.log_dict(metrics, 'input_cfg/metric_names.json')

def run_training(model, data_loader):

    # load generators
    gen_train = data_loader.get_generator(primary_set = True)
    gen_val = data_loader.get_generator(primary_set = False)

    # datasets from generators
    input_shape = ((33, 33, 5), None)
    input_types = (tf.float32, tf.float32)
    data_train = tf.data.Dataset.from_generator(
        gen_train, output_types = input_types, output_shapes = input_shape
        ).prefetch(tf.data.AUTOTUNE).batch(data_loader.n_tau).take(data_loader.n_batches)
    data_val = tf.data.Dataset.from_generator(
        gen_val, output_types = input_types, output_shapes = input_shape
        ).prefetch(tf.data.AUTOTUNE).batch(data_loader.n_tau).take(data_loader.n_batches_val)

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
        with open(to_absolute_path(f'{cfg.path_to_mlflow}/{run_kwargs["experiment_id"]}/{run_id}/artifacts/model_summary.txt')) as f:
            for l in f:
                if (s:='Trainable params: ') in l:
                    mlflow.log_param('n_train_params', int(l.split(s)[-1].replace(',', '')))

        # log training related files
        mlflow.log_dict(training_cfg, 'input_cfg/training_cfg.yaml')
        mlflow.log_artifact(to_absolute_path("TrainingNN.py"), 'input_cfg')

        # log hydra files
        mlflow.log_artifacts('.hydra', 'input_cfg/hydra')
        mlflow.log_artifact('TrainingNN.log', 'input_cfg/hydra')

        # log misc. info
        mlflow.log_param('run_id', run_id)
        print(f'\nTraining has finished! Corresponding MLflow experiment name (ID): {cfg.experiment_name}({run_kwargs["experiment_id"]}), and run ID: {run_id}\n')
        mlflow.end_run()


if __name__ == '__main__':
    main()