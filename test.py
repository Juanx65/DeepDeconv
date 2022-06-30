import os
from pathlib import Path
import numpy as np
import scipy.fft
import scipy.signal
from scipy.signal import windows
import matplotlib.pyplot as plt
import h5py
from models import UNet
from models import CallBacks
from random import choice
import tensorflow as tf
import argparse
from scipy.io import loadmat
from random import randint

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'

def test(opt):

    """ Load DAS data """
    annots = loadmat('data/data_deltas.mat')

    data = np.array(annots["array_output"])
    new_data = []
    for _, dato in enumerate(data):
        phase = randint(0,42)
        temp_dato = np.zeros((24, 1024))
        #primer canal
        temp_dato[0] = dato
        #siguientes canales
        for ch in range(1,23):
            dato  = fill_channel(dato, (ch+1)*phase)
            temp_dato[ch] = dato
        new_data.append(temp_dato)
    new_data = np.array(new_data)
    data = new_data

    """ Load impulse response """
    kernel = np.array(annots["chirp_kernel"][0])
    kernel = np.flip(kernel) #ya no se necesita flipiar, al parecer

    """ Some model parameters """
    _, Nch, Nt = data.shape
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

    """ CARGAR PESOS AL MODELO """
    model.load_weights(str(str(Path(__file__).parent) + opt.weights)).expect_partial()#'/checkpoints/cp-0100.ckpt'))

    x = data

    """ FINALMENTE HACER LA PRUEBA"""
    i = input("index data show: ")
    while True:
        if i != "":
            image_index = int(i)
            x_hat, y_hat = model.call(x[image_index][None,:,:])
            x_hat = tf.reshape(x_hat,[24,1024])
            y_hat = tf.reshape(y_hat,[24,1024])

            """ GRAFICAR LOS RESULTADOS """
            samp = 80.
            t = np.arange(x_hat.shape[1]) / samp

            f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
            ax1.set_title('X Original (integrado)')
            ax2.set_title('X_hat')
            ax3.set_title('Y_hat')

            f.suptitle('DATA'+ str(i), fontsize=16)
            #subplot1: origina
            for i, wv in enumerate(x[image_index]):
                ax1.plot( t, wv - 8 * i, "tab:orange")
                #break
            plt.tight_layout()
            plt.grid()

            #subplot2: x_hat-> estimación de la entrada (conv kernel con la salida)
            for i, wv in enumerate(x_hat):
                ax2.plot(t,(wv - 8 * i), "tab:red")
                #break
            plt.tight_layout()
            plt.grid()

            #subplot3: y_hat->
            for i, wv in enumerate(y_hat):
                ax3.plot(t,wv - 8 * i, c="k")
                #break
            plt.tight_layout()
            plt.grid()

            #plt.savefig("figures/multi_cars_impulse.pdf")
            plt.grid()
            plt.show()
            plt.close()
        else:
            break
        i = input("index data show: ")

## desfase de data para cada canal de manera circular
## lista: dato Original
## idx: indice a partir de donde se copia del primer dato
def fill_channel(lista,idx):
    channel =np.zeros(len(lista))

    for i  in range(len(lista[idx:])):
        channel[idx + i] = lista[i]


    return channel


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default = "data",type=str,help='dir to the dataset')
    parser.add_argument('--kernel_dir', default = "kernels",type=str,help='dir to the dataset')
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
