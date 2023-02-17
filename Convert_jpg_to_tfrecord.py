import tensorflow  as tf
import numpy as np
from utils import *
from models import *
import glob
from PIL import Image
import os
import tempfile

print('Start...')
path = 'flower_photos/all/'
train_filenames=(tf.io.gfile.glob(path+'*.jpg'))

print(len(train_filenames))

#########################
#  Convert jpg to tfrecord files
#########################
def write_tfrecords(x, filename):
    writer = tf.io.TFRecordWriter(filename)

    for path in x:
        print( path)
        img = Image.open(path)
        img = np.array(img.resize((64,64)))
        
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
            }))
        writer.write(example.SerializeToString())

if __name__ == '__main__':


    write_tfrecords(train_filenames , 'flowers.tfrecords')