import os
import time
import random
import logging
import warnings
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from Model_NoDef import *

random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Feature
FEATURE = "tiktok"
GPU_ID = "0"
LENGTH = 5000
NB_CLASSES = 100

# Hyperparameters
NB_EPOCH = 30       # original is 30
BATCH_SIZE = 128
VERBOSE = 2
OPTIMIZER = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
INPUT_SHAPE = (LENGTH,1)

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# Variables for logs
CURRENT_DIR = os.path.dirname(__file__)
LOG_BASE_DIR = os.path.join(CURRENT_DIR, 'logs')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOG_BASE_DIR, f'df_{FEATURE}_{TIMESTAMP}.txt')
LOG_TB_FILE = os.path.join(LOG_BASE_DIR, f'df_tb_{FEATURE}_{TIMESTAMP}')


class LoggerCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        
        log_message(f"Epoch {epoch+1}: "
                    f"loss={loss:.4f}, accuracy={accuracy:.4f}, "
                    f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")


def setup_logging():
    if not os.path.exists(LOG_BASE_DIR):
        os.makedirs(LOG_BASE_DIR)
    
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    tb_callback = TensorBoard(log_dir=LOG_TB_FILE, histogram_freq=1)
    return tb_callback


def log_message(message, level='info'):
    if level == 'info':
        logging.info(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'error':
        logging.error(message)
    else:
        logging.debug(message)


def parse_args():
    global FEATURE, NB_EPOCH, BATCH_SIZE, GPU_ID, VERBOSE, LENGTH, NB_CLASSES
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", '--feature', type=str, default=FEATURE) 
    parser.add_argument("-e", '--nb_epoch', type=int, default=NB_EPOCH) # max_size: -1 입력
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("-g", '--gpu_id', type=str, default=GPU_ID)
    parser.add_argument("-v", "--verbose", type=int, default=VERBOSE)
    parser.add_argument("--length", type=int, default=LENGTH)
    parser.add_argument("--nb_classes", type=int, default=NB_CLASSES)
    args = parser.parse_args()

    FEATURE = args.feature
    NB_EPOCH = args.nb_epoch
    BATCH_SIZE = args.batch_size
    GPU_ID = args.gpu_id
    VERBOSE = args.verbose
    LENGTH = args.length
    NB_CLASSES = args.nb_classes
    return args


def load_data(feature):
    feature_dict = {
        "tiktok": f"{os.path.join(CURRENT_DIR, 'dataset', 'tiktok.npz')}"
		# Adde more feature .npz here
    }

    npz_path = feature_dict.get(feature)
    data = np.load(npz_path)
    features, labels = data['data'], data['labels']
    log_message(f"Data shapes: {features.shape}, {labels.shape}")

    # split 8:1:1
    x_train, x_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    K.set_image_data_format("channels_first")

    # Convert data as float32 type
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('float32')
    y_valid = y_valid.astype('float32')
    y_test = y_test.astype('float32')

    # we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
    x_train = x_train[:, :,np.newaxis]
    x_valid = x_valid[:, :,np.newaxis]
    x_test = x_test[:, :,np.newaxis]

    num_classes = len(np.unique(y_train))
    log_message(f"Number of classes: {num_classes}")
    log_message(f"Train data shapes: {x_train.shape}, {y_train.shape}")
    log_message(f"Valid data shapes: {x_valid.shape}, {y_valid.shape}")
    log_message(f"Test data shapes: {x_test.shape}, {y_test.shape}")

    y_train = to_categorical(y_train, NB_CLASSES)
    y_valid = to_categorical(y_valid, NB_CLASSES)
    y_test = to_categorical(y_test, NB_CLASSES)

    return x_train, y_train, x_valid, y_valid, x_test, y_test, num_classes

    
def main():
    tb_callback = setup_logging()
    logger_callback = LoggerCallback()

    args = parse_args()
    log_message(f"Parsed arguments: {args}")
    
    # data loading
    x_train, y_train, x_valid, y_valid, x_test, y_test, num_classes = load_data(FEATURE)
    
    # set the model
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)


    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
    log_message("Model compiled")

    # Start training
    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                        verbose=VERBOSE, validation_data=(x_valid, y_valid),
                        callbacks=[tb_callback, logger_callback])

    # Start evaluating model with testing data
    score_test = model.evaluate(x_test, y_test, verbose=VERBOSE)
    log_message(f"Testing accuracy: {score_test[1]}")


if __name__ == "__main__":
    main()