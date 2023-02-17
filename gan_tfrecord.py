


import os
import numpy as np
from glob import glob
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils import *
from models import *
from tensorflow.keras.callbacks import TensorBoard 
import argparse

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='GAN')
    parser.add_argument("--epochs",dest= 'epochs', default= 100) 
    parser.add_argument("--batch_size",dest= 'batch_size', default= 32) 
    parser.add_argument("--learning_rate",dest= 'learning_rate', default= 0.0001) 
    parser.add_argument("--checkpoints_dir",dest= 'checkpoints_dir', default= 'ckpt')
    parser.add_argument("--IMG_H",dest= 'IMG_H', default= 64) 
    parser.add_argument("--IMG_W",dest= 'IMG_W', default= 64) 
    parser.add_argument("--IMG_C",dest= 'IMG_C', default= 3) 
    parser.add_argument("--tfrecord_dir",dest= 'tfrecord_dir', default= 'tfrecord files')
    parser.add_argument("--latent_dim",dest= 'latent_dim', default= 200) 
    return parser.parse_args()


###  python -m tensorboard.main --logdir=logs/

if __name__ == "__main__":
    ## Hyperparameters
    args = arg_parse()
    batch_size = args.batch_size
    latent_dim = args.latent_dim
    num_epochs = args.epochs
    images_path = glob(args.tfrecord_dir + "/*")
    d_model = build_discriminator(args)
    g_model = build_generator(args)
    # d_model.load_weights("saved_model/d_model.h5")
    # g_model.load_weights("saved_model/g_model.h5")
    d_model.summary()
    g_model.summary()

    gan = GAN(d_model, g_model, latent_dim)
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    # d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5)
    # g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5)

    d_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5)

    gan.compile(d_optimizer, g_optimizer, bce_loss_fn)
    filenames = 'flowers.tfrecords'
    images_dataset = read_dataset(filenames, args.batch_size)
    

    ###   Tensorboard activation
    filename_tensorbaord = 'my_saved_model'
    tensorboard = TensorBoard(log_dir = 'logs\\{}'.format(filename_tensorbaord))


    # for epoch in range(num_epochs):
    # gan.fit(images_dataset, epochs=1, callbacks= tensorboard)
    gan.fit(images_dataset, epochs=200, callbacks= tensorboard)
    # g_model.save("saved_model/g_model.h5")
    # d_model.save("saved_model/d_model.h5")
    n_samples = 25
    noise = np.random.normal(size=(n_samples, latent_dim))
    examples = g_model.predict(noise)
    save_plot(examples, 1, int(np.sqrt(n_samples)))

    
