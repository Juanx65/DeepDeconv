import numba as nb
import numpy as np
from spectrum import dpss
from scipy.fft import rfft, rfftfreq


class Beamformer:
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        pass
    
    def construct_slowness_grid(self):
        
        Nslow = self.Nslow
        if Nslow % 2 == 0:
            Nslow += 1
            self.Nslow = Nslow
        
        slow_max = self.slow_max
        slow_grid = np.linspace(-slow_max, slow_max, Nslow)
        self.slowness = slow_grid
        
        return None
    
    def construct_times(self):
        
        Nch = self.Nch
        gauge = self.gauge
        x = np.arange(Nch) * gauge
        x -= x.mean()
        
        Nslow = self.Nslow
        slowness = self.slowness
        
        dt = np.outer(slowness, x)
        
        self.dt = dt
        return dt
    
    def precompute_A(self, f):
        dt = self.dt
        A = np.exp(-2j * np.pi * f * dt)
        self.A = A
        return A
    
    @staticmethod
    @nb.njit(nogil=True, parallel=True)
    def compute_Cxy_jit(Xf, weights, scale):
        tapers, Nch, Nt = Xf.shape
        Cxy = np.zeros((Nch, Nch, Nt), dtype=nb.complex64)
        Xfc = Xf.conj()

        for i in nb.prange(Nch):
            for j in nb.prange(i + 1):
                # Do not prange this one!
                for k in range(tapers):
                    Cxy[i, j] += weights[k] * Xf[k, i] * Xfc[k, j]
                Cxy[i, j] = Cxy[i, j] * (scale[i] * scale[j])

        return Cxy

    def CMTM(self, X, Nw, freq_band=None, fsamp=None, scale=True, jit=False):

        # Number of tapers
        K = int(2 * Nw)
        # Number of stations (m), time sampling points (Nx)
        m, Nf = X.shape
        # Next power of 2 (for FFT)
        NFFT = 2**int(np.log2(Nf) + 1) + 1

        # Subtract mean (over time axis) for each station
        X_mean = np.mean(X, axis=1)
        X_mean = np.tile(X_mean, [Nf, 1]).T
        X = X - X_mean

        # Compute taper weight coefficients
        tapers, eigenvalues = dpss(N=Nf, NW=Nw, k=K)

        # Compute weights from eigenvalues
        weights = eigenvalues / (np.arange(K) + 1).astype(float)

        # Align tapers with X
        tapers = np.tile(tapers.T, [m, 1, 1])
        tapers = np.swapaxes(tapers, 0, 1)

        # Compute tapered FFT of X
        # Note that X is assumed to be real, so that the negative frequencies can be discarded
        Xf = rfft(np.multiply(tapers, X), 2 * NFFT, axis=-1)

        # Multitaper power spectrum (not scaled by weights.sum()!)
        Pk = np.abs(Xf)**2
        Pxx = np.sum(Pk.T * weights, axis=-1).T
        inv_Px = 1 / np.sqrt(Pxx)

        inv_sum_weights = 1.0 / weights.sum()

        # If a specific frequency band is given
        if freq_band is not None:
            # Check if the sampling frequency is specified
            if fsamp is None:
                print("When a frequency band is selected, fsamp must be provided")
                return False
            # Compute the frequency range
            freqs = rfftfreq(n=2 * NFFT, d=1.0 / fsamp)
            # Select the frequency band indices
            inds = (freqs >= freq_band[0]) & (freqs < freq_band[1])
            # Slice the vectors
            Xf = Xf[:, :, inds]
            inv_Px = inv_Px[:, inds]

        # Buffer for covariance matrix
        if jit:
            Ns = Xf.shape[1]
            if scale:
                # Vector for scaling
                scale_vec = inv_Px
                # Compute covariance matrix
                Cxy = self.compute_Cxy_jit(Xf, weights, scale_vec)
                # Make Cxy Hermitian
                Cxy = Cxy + np.transpose(Cxy.conj(), axes=[1, 0, 2])
                # Add ones to diagonal
                for i in range(Ns):
                    Cxy[i, i] = 1
            else:
                # Vector for scaling
                scale_vec = np.sqrt(np.ones(Ns) * inv_sum_weights / Xf.shape[2])
                # Compute covariance matrix
                Cxy = self.compute_Cxy_jit(Xf, weights, scale_vec)
                # Make Cxy Hermitian
                Cxy = Cxy + np.transpose(Cxy.conj(), axes=[1, 0, 2])
                # Correct diagonal
                for i in range(Ns):
                    Cxy[i, i] *= 0.5

        else:
            Cxy = np.zeros((m, m, Xf.shape[2]), dtype=complex)

            # Loop over all stations
            for i in range(m):
                # Do only lower triangle
                for j in range(i):
                    # Compute SUM[w_k . X_k . Y*_k] using Einstein notation
                    Pxy = np.einsum("k,kt,kt->t", weights, Xf[:, i], Xf[:, j].conj(), optimize=True)
                    # Store result in covariance matrix
                    if scale:
                        Cxy[i, j] = Pxy * (inv_Px[i] * inv_Px[j])
                    else:
                        Cxy[i, j] = Pxy * inv_sum_weights / Xf.shape[2]
                if not scale:
                    Cxy[i, i] = 0.5 * np.einsum("k,kt,kt->t", weights, Xf[:, i], Xf[:, i].conj(), optimize=True) * inv_sum_weights / Xf.shape[2]
            # Make Cxy Hermitian
            Cxy = Cxy + np.transpose(Cxy.conj(), axes=[1, 0, 2])
            # Add ones to diagonal
            if scale:
                Cxy = Cxy + np.tile(np.eye(m), [Cxy.shape[2], 1, 1]).T
        return Cxy

    def noise_space_projection(self, Rxx, sources=3, mode="MUSIC"):
        # Number of source locations (Nx, Ny), number of stations (m)
        Nslow = self.Nslow
        A = self.A
        _, m = A.shape
        scale = 1.0 / m

        # Total projection onto noise space
        Pm = np.zeros(Nslow, dtype=complex)
        
        Rxx = np.nanmean(Rxx, axis=-1)
        
        # Traditional beamforming: maximise projection of steering vector onto covariance matrix
        if mode == "beam":
            # Cast to complex, which dramatically reduces overhead. No clue why, because Rxx is already complex...
            Un = Rxx.astype(complex)
            # Project steering vector onto subspace
            Pm = np.einsum("sn, nk, sk->s", A.conj(), Un, A, optimize=True)
        # MUSIC: minimise projection of steering vector onto noise space
        elif mode == "MUSIC":
            # Compute eigenvalues/vectors assuming Rxx is complex Hermitian (conjugate symmetric)
            # Eigenvalues appear in ascending order
            l, v = np.linalg.eigh(Rxx)
            M = sources
            # Extract noise space (size n-M)
            # NOTE: in original code, un was labelled "signal space"!
            un = v[:, :m - M]
            # Precompute un.un*
            Un = np.dot(un, un.conj().T)
            # Project steering vector onto subspace
            Pm = np.einsum("sn, nk, sk->s", A.conj(), Un, A, optimize=True)
        else:
            print("Mode '%s' not recognised. Aborting...")
            return

        return np.real(Pm) * scale