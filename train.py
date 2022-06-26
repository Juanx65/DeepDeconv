import os
from pathlib import Path
import numpy as np
import scipy.fft
from scipy.signal import windows
import matplotlib.pyplot as plt
import h5py
from models import UNet
from models import CallBacks

import tensorflow as tf
import argparse
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'

def train(opt):
    """ Variables necesarias """

    cwd = os.getcwd()
    datadir = os.path.join(cwd, opt.data_dir)
    data_file = os.path.join(datadir, "DAS_data.h5")

    buf = 100_000
    samp = 50.

    epochs = opt.epochs
    batch_size = opt.batch_size

    """ Load DAS data """
    # DAS_data.h5 -> datos para leer (1.5GB) strain rate -> hay que integrarlos
    with h5py.File(data_file, "r") as f:
         # Nch : numero de canales, Nt = largo de muestras (1024?), SI
        Nch, Nt = f["strainrate"].shape
        split = int(0.9 * Nt)
        data = f["strainrate"][:, split:-buf].astype(np.float32)
    # se normaliza cada trace respecto a su desviación estandar
    data /= data.std()
    Nch, Nt = data.shape
    # Shape: 24 x 180_000 (son 24 sensores/canales, y 180_000 muestras?)
    data_light = data[:, :int(3600 * samp)] # light traffic: lo que mencionan en el paper
    data_heavy = data[:, 440_000:int(440_000 + 3600 * samp)] # heavy traffic


    """ Integrate DAS data (strain rate -> strain) """
    win = windows.tukey(Nt, alpha=0.1)
    freqs = scipy.fft.rfftfreq(Nt, d=1/samp)
    Y = scipy.fft.rfft(win * data, axis=1)
    Y_int = -Y / (2j * np.pi * freqs)
    Y_int[:, 0] = 0
    data_int = scipy.fft.irfft(Y_int, axis=1)
    data_int /= data_int.std()

    data_int_light = data_int[:, :int(3600 * samp)]
    data_int_heavy = data_int[:, 440_000:int(440_000 + 3600 * samp)]


    """ Load impulse response """
    kernel = np.load(os.path.join(datadir, "kernel.npy"))
    # Se normaliza el kernel respecto al máximo (a diferencia de las traces DAS que se normalizan respecto a la desviación estandar)
    kernel = kernel / kernel.max()

    """ Some model parameters """
    rho = 10.0
    f0 = 8
    blocks = 3
    noise = opt.dropout
    deep_win = opt.deep_win

    """ Init Deep Learning model """
    model = UNet(
        kernel.astype(np.float32), lam=rho, f0=f0,
        data_shape=(Nch, deep_win, 1), blocks=blocks, AA=False, bn=False, dropout=noise
    )

    model.construct()
    model.compile()

    ############################################################################################
    """ Formatear data para entrenar """
    # Number of chunks
    if opt.data_heavy:
        data_int = data_int_heavy
    else:
        data_int = data_int_light

    Nwin = data_int.shape[1] // deep_win
    # Total number of time samples to be processed
    Nt_deep = Nwin * deep_win

    """ Mould data into right shape for UNet """

    ########################################################
    data_split = np.stack(np.split(data_int[:, :Nt_deep], Nwin, axis=-1), axis=0)
    data_split = np.stack(data_split, axis=0)
    data_split = np.expand_dims(data_split, axis=-1)


    # Buffer for impulses
    x = np.zeros_like(data_split)
    N = data_split.shape[0] // batch_size
    r = data_split.shape[0] % batch_size
    #N = data_split.shape[0] // 1
    #r = data_split.shape[0] % 1

    #win2 = windows.tukey(batch_size, alpha=0.1)

    #for i in range(N):
    #    n_slice = slice(i * 1, (i + 1) * 1)
    #    x_i = data_split[n_slice]
    #    algo = np.array(x_i)
    #    algo2 = DataGenerator(algo,win2,batch_size,batch_size)
    #    print(len(algo2))
    #    x[i] =algo2
    # Loop over chunks#

    for i in range(N):
        n_slice = slice(i * batch_size, (i + 1) * batch_size)
        x_i = data_split[n_slice]
        x[n_slice] = x_i
    # If there is some residual chunk: process that too
    if r > 0:
        n_slice = slice((i + 1) * batch_size, None)
        x_i = data_split[n_slice]
        x[n_slice] = x_i

    #impulses_deep = np.concatenate(np.squeeze(x), axis=1)

    checkpoint_filepath = str(str(Path(__file__).parent) +opt.checkpoint)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=True,
        update_freq="epoch")
    history = model.fit(
        x,
        validation_split=0.5,
        epochs=epochs,
        callbacks=[model_checkpoint_callback],
        batch_size=batch_size
    )

    """ Printear algunas cosas para ver como se entreno """
    acc = history.history['l1']
    val_acc = history.history['val_l1']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)#epochs

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 32, type=int,help='batch size to train')
    parser.add_argument('--epochs', default = 100 ,type=int,help='epoch to train')
    parser.add_argument('--data_dir', default = "data",type=str,help='dir to the dataset')
    parser.add_argument('--data_heavy', default = True,type=bool,help='type of data to train, if True, it will traing with the heavy data')
    parser.add_argument('--checkpoint', default = "/checkpoints/best.ckpt",type=str,help='dir to save the weights og the training')
    parser.add_argument('--logs', default = "logs/",type=str,help='dir to save the logs of the training')
    parser.add_argument('--optimizer', default = 'adam',type=str,help='optimizer for the model ej: adam, sgd, adamax ...')
    parser.add_argument('--dropout', default = 1.0,type=float,help='% dropout to use')
    parser.add_argument('--deep_win', default = 1024,type=int,help='Number of samples per chunk')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):
	train(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
