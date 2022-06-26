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
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'

def test(opt):
    cwd = os.getcwd()
    datadir = os.path.join(cwd, opt.data_dir)
    data_file = os.path.join(datadir, "DAS_data.h5")
    buf = 100_000
    samp = 50.
    """ CARGAR EL MODELO """
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

    """ CARGAR DATA PARA PRUEBAS """

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


    """ Init Deep Learning model """
    model = UNet(
        kernel.astype(np.float32), lam=rho, f0=f0,
        data_shape=(Nch, deep_win, 1), blocks=blocks, AA=False, bn=False, dropout=noise
    )

    model.construct()
    model.compile()

    """ CARGAR PESOS AL MODELO """
    model.load_weights(str(str(Path(__file__).parent) + opt.weights)).expect_partial()#'/checkpoints/cp-0100.ckpt'))

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
    ########################################################

    """VAMOS A PROBAR CON DATA HEAVY"""
    data_int = data_int_light
    #
    Nwin = data_int.shape[1] // deep_win
    # Total number of time samples to be processed
    Nt_deep = Nwin * deep_win
    #
    data_split = np.stack(np.split(data_int[:, :Nt_deep], Nwin, axis=-1), axis=0)
    data_split = np.stack(data_split, axis=0)
    data_split = np.expand_dims(data_split, axis=-1)
    # Buffer for impulses
    batch_size = 1 # PARA TENER SOLO UN DATO EN 1 BATCH

    x = np.zeros_like(data_split)
    N = data_split.shape[0] // batch_size
    r = data_split.shape[0] % batch_size
    for i in range(N):
        n_slice = slice(i * batch_size, (i + 1) * batch_size)
        x_i = data_split[n_slice]
        x[n_slice] = x_i
    # If there is some residual chunk: process that too
    if r > 0:
        n_slice = slice((N-1 + 1) * batch_size, None)
        x_i = data_split[n_slice]
        x[n_slice] = x_i

    """ FINALMENTE HACER LA PRUEBA"""
    image_index = 174   # (numero de imagen dentro del batch) el maximo es 174 (data light o heavy)
    x_hat, y_hat = model.call(x[image_index][None,:,:,:])
    x_hat = tf.reshape(x_hat,[24,1024])
    y_hat = tf.reshape(y_hat,[24,1024])
    #print(x_hat, x_hat.shape)

    """ GRAFICAR LOS RESULTADOS """
    samp = 80.

    t = np.arange(x_hat.shape[1]) / samp
    fig = plt.figure(figsize=(9, 3))
    gs = fig.add_gridspec(1, 3)


    #subplot1 ?
    ax = fig.add_subplot(gs[0,1])
    ax.set_xlim((t.min(), t.max()))
    ax.set_xlabel("time [s]")



    for i, wv in enumerate(x[image_index]):
        ax.plot(t, wv - 8 * i, "tab:orange")
        break

    for letter, ax in zip("ab", fig.axes):
        ax.set_yticks([])
        #ax.text(x=0.0, y=1.0, transform=ax.transAxes, s=letter, **letter_params)
        for spine in ("left", "top", "right"):
            ax.spines[spine].set_visible(False)
    plt.tight_layout()

    #subplot2 ?

    for i, wv in enumerate(x_hat):
        ax.plot(t, wv - 8 * i, "tab:red")
        break


    for letter, ax in zip("ab", fig.axes):
        ax.set_yticks([])
        #ax.text(x=0.0, y=1.0, transform=ax.transAxes, s=letter, **letter_params)
        for spine in ("left", "top", "right"):
            ax.spines[spine].set_visible(False)
    plt.tight_layout()

    #subplot3 ?
    for i, wv in enumerate(y_hat):
        ax.plot(t, wv - 8 * i, c="k")
        break

    for letter, ax in zip("ab", fig.axes):
        ax.set_yticks([])
        #ax.text(x=0.0, y=1.0, transform=ax.transAxes, s=letter, **letter_params)
        for spine in ("left", "top", "right"):
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    #plt.savefig("figures/multi_cars_impulse.pdf")
    plt.show()

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default = "data",type=str,help='dir to the dataset')
    parser.add_argument('--weights',default = '/checkpoints/best.ckpt', type=str,help='load weights path')
    parser.add_argument('--optimizer', default = 'adam',type=str,help='optimizer for the model ej: adam, sgd, adamax ...')
    parser.add_argument('--dropout', default = 1.0,type=float,help='% dropout to use')
    parser.add_argument('--deep_win', default = 1024,type=int,help='Number of samples per chunk')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):
	test(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
