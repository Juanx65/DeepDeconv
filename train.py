import os
from pathlib import Path
import numpy as np
import scipy.fft
from scipy.signal import windows
import matplotlib.pyplot as plt
import h5py
from models import UNet
from models import CallBacks
from random import choice
import tensorflow as tf
import argparse
from models import DataGenerator
from datetime import datetime
from scipy.io import loadmat
import json
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'

def train(opt):

    """ Variables necesarias """
    cwd = os.getcwd()
    #datadir = os.path.join(cwd, opt.data_dir)
    #kerneldir =  os.path.join(cwd, opt.kernel_dir)
    #data_file = os.path.join(datadir, "DAS_data.h5")

    samp = 50.

    epochs = opt.epochs
    batch_size = opt.batch_size

    """ Load DAS data """
    # DAS_data.h5 -> datos para leer (1.5GB) strain rate -> hay que integrarlos
    annots = loadmat('data/data_positive_deltas.mat')
    data = np.array(annots["array_output"])

    new_data = []
    for d, dato in enumerate(data):
        temp_dato = np.zeros((24, 1024))
        for ch in range(23):
            temp_dato[ch] = dato
        new_data.append(temp_dato)
    new_data = np.array(new_data)
    data = new_data

    data = data.reshape(10000,24,1024,1)

    _, Nch, Nt,_ = data.shape
    # Shape: 24 x 180_000 (son 24 sensores/canales, y 180_000 muestras?)

    ########################################
    """ Load impulse response """
    kernel = np.array(annots["chirp_kernel"][0])
    kernel = np.flip(kernel)
    """ Some model parameters """
    rho = 10.0
    f0 = 8
    blocks = 3
    dropout_value = opt.dropout
    deep_win = opt.deep_win

    """ Init Deep Learning model """
    model = UNet(
        kernel.astype(np.float32), lam=rho, f0=f0,
        data_shape=(Nch, deep_win, 1), blocks=blocks, AA=False, bn=False, dropout=dropout_value
    )

    model.construct()
    model.compile()
    #data = tf.reshape(data,[10000, 24,1024, 1])
    checkpoint_filepath = str(str(Path(__file__).parent) +opt.checkpoint)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=True,
        update_freq="epoch")
    history = model.fit(
        data,
        validation_split=0.5,
        epochs=epochs,
        callbacks=[model_checkpoint_callback],
        batch_size=batch_size
    )


    timeID = datetime.now()
    timeString = timeID.strftime("%Y-%m-%d_%H-%M-%S")
    file_prefix = 'train_{}.json'.format(timeString)

    with open(os.path.join('trainHistory/',file_prefix), 'w') as file:
        json.dump(history.history, file)

    """ Printear algunas cosas para ver como se entreno """
    loss1 = history.history['l1']
    val_loss1 = history.history['val_l1']

    loss2 = history.history['l2']
    val_loss2 = history.history['val_l2']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)#epochs

    plt.figure(figsize=(8, 8))

    plt.subplot(3, 1, 1)
    plt.plot(epochs_range, loss1, label='Training Sparsity (L1)')
    plt.plot(epochs_range, val_loss1, label='Validation Sparsity (L1)')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Sparsity')

    plt.subplot(3, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Total Loss')

    plt.subplot(3, 1, 3)
    plt.plot(epochs_range, loss2, label='Training Loss (L2)')
    plt.plot(epochs_range, val_loss2, label='Validation Loss (L2)')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Loss (L2)')

    plt.show()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 128, type=int,help='batch size to train')
    parser.add_argument('--epochs', default = 200 ,type=int,help='epoch to train')
    parser.add_argument('--data_dir', default = "data",type=str,help='dir to the dataset')
    parser.add_argument('--kernel_dir', default = "kernels",type=str,help='dir to the dataset')
    parser.add_argument('--checkpoint', default = "/checkpoints/best.ckpt",type=str,help='dir to save the weights og the training')
    parser.add_argument('--dropout', default = 1.0,type=float,help='percentage dropout to use')
    parser.add_argument('--deep_win', default = 1024,type=int,help='Number of samples per chunk')

    opt = parser.parse_args()
    return opt

def main(opt):
    train(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
