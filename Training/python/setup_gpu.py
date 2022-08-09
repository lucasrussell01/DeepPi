import tensorflow as tf

def setup_gpu(gpu_cfg): # from DeepTau
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_cfg['gpu_index']], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(gpus[gpu_cfg['gpu_index']],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_cfg['gpu_mem']*1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)