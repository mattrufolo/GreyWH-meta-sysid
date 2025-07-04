import math
import torch
import numpy as np
from numpy import empty
from torch.utils.data import DataLoader, IterableDataset
import CSTR_functions as CSTR_F
from lti import drss_matrices, dlsim, nn_fun


def generate_input(rng, batch_size=1, N=1000, p_low_pass=0.5, p_high_pass=0.05):
    """
    Generate input signals with specified filtering characteristics.

    Parameters:
    batch_size (int): Number of signal batches to generate. Each batch contains one signal,
                      enabling the simultaneous generation of multiple signals.

    N (int): Total number of data points in each signal. This defines both the length of the
             signal in the time domain and the resolution in the frequency domain.

    p_low_pass (float): Probability (range 0 to 1) that the minimum frequency will be set to zero,
                        effectively applying a low-pass filter.

    p_high_pass (float): Probability (range 0 to 1) that the maximum frequency will be set to the
                         Nyquist frequency, effectively applying a high-pass filter.

    Returns:
    u (ndarray): Array of generated time-domain signals. The array shape is (batch_size, N),
                 where each row represents a single time-domain signal.

    uf (ndarray): Array of frequency-domain representations of the signals. This complex array
                  corresponds to the frequency components of the signals in 'u'.

    fmin (int): Minimum frequency index that is active in the frequency domain representation,
                defining the lower bound of the filter's passband.

    fmax (int): Maximum frequency index that is active, defining the upper bound of the filter's
                passband.
    """

    # Randomly select two frequencies within the valid range
    f1, f2 = rng.integers(size=2, low=1, high=N // 2 + 1)
    # Determine minimum and maximum frequencies based on random selection
    fmin, fmax = sorted([f1, f2])

    # Apply low-pass filter with the probability p_low_pass
    if rng.uniform() < p_low_pass:
        fmin = 1
    # Apply high-pass filter with the probability p_high_pass
    if rng.uniform() < p_high_pass:
        fmax = N // 2

    # Create an array of zeros for frequency components
    uf = np.zeros((batch_size, N // 2 + 1), dtype=np.complex64)
    # Assign random phase shifts to the frequency components within the passband
    uf[:, fmin:fmax + 1] = np.exp(1j * rng.uniform(low=0, high=2 * np.pi, size=(batch_size, fmax - fmin + 1)))

    # Inverse real FFT to convert frequency domain to time domain
    u = np.fft.irfft(uf)
    # Normalize the signal
    u *= np.sqrt((N // 2 + 1) / (fmax - fmin + 1) * N)

    return u, uf, fmin, fmax





class BenchmarkDataset(IterableDataset):
    def __init__(self,seq_len_ctx,seq_len_new,seq_len_n_in,system_seed,input_seed,noise_seed,normalize=True):
        super(BenchmarkDataset).__init__()

        self.seq_len_ctx = seq_len_ctx
        self.seq_len_new = seq_len_new
        self.seq_len_n_in = seq_len_n_in
        self.system_seed = system_seed
        self.input_seed = input_seed
        self.noise_seed = noise_seed
        self.normalize = normalize
        self.system_rng = np.random.default_rng(system_seed)  # source of randomness for system generation
        self.input_rng = np.random.default_rng(input_seed)  # source of randomness for input generation
        self.noise_rng = np.random.default_rng(noise_seed)  # source of randomness for noise generation
        self.train_val, self.test  = nonlinear_benchmarks.Silverbox()#nonlinear_benchmarks.WienerHammerBenchMark()
        # print(test.state_initialization_window_length) # = 4
        
    def __iter__(self):   

        while True: 
            (train_val_u, train_val_y) = self.train_val
            _,(test_u, test_y),_ = self.test
            idx_train = int(self.system_rng.uniform(0,train_val_u.shape[0]-self.seq_len_ctx))
            idx_test = int(self.system_rng.uniform(0,test_u.shape[0]-self.seq_len_new-self.seq_len_n_in))
            u_ctx,y_ctx = train_val_u[idx_train:idx_train+self.seq_len_ctx],train_val_y[idx_train:idx_train+self.seq_len_ctx]
            u_new,y_new = test_u[idx_test:idx_test+self.seq_len_new+self.seq_len_n_in],test_y[idx_test:idx_test+self.seq_len_new+self.seq_len_n_in]

            # u = torch.concat([torch.tensor(u_ctx),torch.tensor(u_new)], axis =0).unsqueeze(1)
            # y = torch.concat([torch.tensor(y_ctx),torch.tensor(y_new)], axis =0).unsqueeze(1)


            # if self.normalize:
            #         u = (u - u.mean(axis=0)) / (u.std(axis=0) + 1e-6)
            #         y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)

            # print(y.shape)
            
            # yield torch.tensor(y), torch.tensor(u)

            u = np.concatenate([u_ctx,u_new], axis =0).reshape([530,1])
            y = np.concatenate([y_ctx,y_new], axis =0).reshape([530,1])


            if self.normalize:
                    u = (u - u.mean(axis=0)) / (u.std(axis=0) + 1e-6)
                    y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)

            # print(y.shape)

            u = u.astype('float32')
            y = y.astype('float32')
            yield torch.tensor(y), torch.tensor(u)

class LinearDynamicalDataset(IterableDataset):
    def __init__(self, nx=5, nu=1, ny=1, random_order=True, seq_len=500,  dtype="float32", normalize=True,
                 system_seed=None, input_seed=None, noise_seed=None,
                **mdlargs):
        super(LinearDynamicalDataset).__init__()
        self.nx = nx

        self.nu = nu
        self.ny = ny
        self.random_order = random_order
        self.seq_len = seq_len
        self.mdlargs = mdlargs # strictly_proper=True, mag_range=(0.5, 0.97), phase_range=(0, math.pi / 2)
        self.dtype = dtype
        self.normalize = normalize
        self.system_seed = system_seed
        self.input_seed = input_seed
        self.noise_seed = noise_seed
        self.system_rng = np.random.default_rng(system_seed)  # source of randomness for system generation
        self.input_rng = np.random.default_rng(input_seed)  # source of randomness for input generation
        self.noise_rng = np.random.default_rng(noise_seed)  # source of randomness for noise generation

    def __iter__(self):

        n_skip = 200

        while True:  # infinite dataset
            # for _ in range(1000):
            G = drss_matrices(states=np.random.randint(1, self.nx+1) if self.random_order else self.nx,
                               inputs=self.nu,
                               outputs=self.ny,
                               rng=self.system_rng,
                               **self.mdlargs)
            #print(G[0])
            #u = np.random.randn(self.seq_len, self.nu)

            u = self.input_rng.normal(size=(self.seq_len + n_skip, 1))

            y = dlsim(*G, u)
            if self.normalize:
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)

            # To have random initial state instead than 0...
            u = u[n_skip:]
            y = y[n_skip:]

            e = self.noise_rng.normal(size=(self.seq_len, 1)) * 0.1 # noise term
            y = y + e

            u = u.astype(self.dtype)
            y = y.astype(self.dtype)

            yield torch.tensor(y), torch.tensor(u)



## put the random order true when we implemented the change in order to deal with variable complexity WH systems
class WHDataset(IterableDataset):
    def __init__(self, nx=5, nu=1, ny=1, seq_len=600, random_order=False,
                 strictly_proper=True, normalize=True, dtype="float32",
                 system_seed=None, input_seed=None, noise_seed=None,
                 **mdlargs):
        super(WHDataset).__init__()
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.seq_len = seq_len
        self.strictly_proper = strictly_proper
        self.dtype = dtype
        self.normalize = normalize
        self.strictly_proper = strictly_proper
        self.random_order = random_order  # random number of states from 1 to nx
        self.mdlargs = mdlargs
        self.system_seed = system_seed
        self.input_seed = input_seed
        self.noise_seed = noise_seed
        self.system_rng = np.random.default_rng(system_seed)  # source of randomness for system generation
        self.input_rng = np.random.default_rng(input_seed)  # source of randomness for input generation
        self.noise_rng = np.random.default_rng(noise_seed)  # source of randomness for noise generation


    def __iter__(self):


        n_in = 1
        n_out = 1
        n_hidden = 29
        n_skip = 200

        while True:  # infinite dataset

            w1 = self.system_rng.normal(size=(n_hidden, n_in)) / np.sqrt(n_in) * 5 / 3
            b1 = self.system_rng.normal(size=(1, n_hidden)) * 1.0
            w2 = self.system_rng.normal(size=(n_out, n_hidden)) / np.sqrt(n_hidden)
            b2 = self.system_rng.normal(size=(1, n_out)) * 1.0

            G1 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                                inputs=1,
                                outputs=1,
                                strictly_proper=self.strictly_proper, # if strictly proper, add 1 delay
                                rng=self.system_rng,
                                **self.mdlargs)

            G2 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                                inputs=1,
                                outputs=1,
                                strictly_proper=False, # not strictly proper, no additional delay
                                rng=self.system_rng,
                                **self.mdlargs)

            # To generate the PRBS input (pseudo random binary signal)
            # choices = [-1.0,1.0]
            # u = CSTR_F.generate_random_binary_signal_rep(choices,20,80,self.seq_len + n_skip, self.input_rng)
            # u = np.array([[i] for i in u])
            # To generate the white noise input signal
            u = self.input_rng.normal(size=(self.seq_len + n_skip, 1))

            # u = generate_input(batch_size=1, N=self.seq_len + n_skip, p_low_pass=0.5, p_high_pass=0.05,
            #                    rng=self.input_rng)[0].reshape(-1, 1)

            # G1
            y1 = dlsim(*G1, u)
            y1 = (y1 - y1[n_skip:].mean(axis=0)) / (y1[n_skip:].std(axis=0) + 1e-6)

            # F
            y2 = nn_fun(y1,w1,b1,w2,b2)

            # G2
            y3 = dlsim(*G2, y2)

            u = u[n_skip:]
            y = y3[n_skip:]

            if self.normalize:
                u = (u - u.mean(axis=0)) / (u.std(axis=0) + 1e-6)
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)

            e = self.noise_rng.normal(size=(self.seq_len, 1)) * 0.01 # noise term
            y = y #+ e

            u = u.astype(self.dtype)
            y = y.astype(self.dtype)

            yield torch.tensor(y), torch.tensor(u)
            
class WHDataset_prbs(IterableDataset):
    def __init__(self, nx=5, nu=1, ny=1, seq_len=600, random_order=True,
                 strictly_proper=True, normalize=True, dtype="float32",
                 system_seed=None, input_seed=None, noise_seed=None,
                 **mdlargs):
        super(WHDataset_prbs).__init__()
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.seq_len = seq_len
        self.strictly_proper = strictly_proper
        self.dtype = dtype
        self.normalize = normalize
        self.strictly_proper = strictly_proper
        self.random_order = random_order  # random number of states from 1 to nx
        self.mdlargs = mdlargs
        self.system_seed = system_seed
        self.input_seed = input_seed
        self.noise_seed = noise_seed
        self.system_rng = np.random.default_rng(system_seed)  # source of randomness for system generation
        self.input_rng = np.random.default_rng(input_seed)  # source of randomness for input generation
        self.noise_rng = np.random.default_rng(noise_seed)  # source of randomness for noise generation


    def __iter__(self):


        n_in = 1
        n_out = 1
        n_hidden = 32
        n_skip = 200

        while True:  # infinite dataset

            w1 = self.system_rng.normal(size=(n_hidden, n_in)) / np.sqrt(n_in) * 5 / 3
            b1 = self.system_rng.normal(size=(1, n_hidden)) * 1.0
            w2 = self.system_rng.normal(size=(n_out, n_hidden)) / np.sqrt(n_hidden)
            b2 = self.system_rng.normal(size=(1, n_out)) * 1.0

            G1 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                                inputs=1,
                                outputs=1,
                                strictly_proper=self.strictly_proper, # if strictly proper, add 1 delay
                                rng=self.system_rng,
                                **self.mdlargs)

            G2 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                                inputs=1,
                                outputs=1,
                                strictly_proper=False, # not strictly proper, no additional delay
                                rng=self.system_rng,
                                **self.mdlargs)

            # To generate the PRBS input (pseudo random binary signal)
            choices = [-1.0,1.0]
            u = CSTR_F.generate_random_binary_signal_rep(choices,20,80,self.seq_len + n_skip, self.input_rng)
            u = np.array([[i] for i in u])
            # To generate the white noise input signal
            # u = self.input_rng.normal(size=(self.seq_len + n_skip, 1))

            # u = generate_input(batch_size=1, N=self.seq_len + n_skip, p_low_pass=0.5, p_high_pass=0.05,
            #                    rng=self.input_rng)[0].reshape(-1, 1)

            # G1
            y1 = dlsim(*G1, u)
            y1 = (y1 - y1[n_skip:].mean(axis=0)) / (y1[n_skip:].std(axis=0) + 1e-6)

            # F
            y2 = nn_fun(y1,w1,b1,w2,b2)

            # G2
            y3 = dlsim(*G2, y2)

            u = u[n_skip:]
            y = y3[n_skip:]

            if self.normalize:
                u = (u - u.mean(axis=0)) / (u.std(axis=0) + 1e-6)
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)

            e = self.noise_rng.normal(size=(self.seq_len, 1)) * 0.01 # noise term
            y = y + e

            u = u.astype(self.dtype)
            y = y.astype(self.dtype)

            yield torch.tensor(y), torch.tensor(u)

def seed_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    dataset.input_rng = np.random.default_rng(dataset.input_seed + 1000*worker_id)
    dataset.system_rng = np.random.default_rng(dataset.system_seed + 1000*worker_id)
    dataset.noise_rng = np.random.default_rng(dataset.noise_seed + 1000*worker_id)


class MultiIterableDataSet(IterableDataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __iter__(self):
        iterators = [iter(dataset) for dataset in self.datasets]

        while True:
            dataset_index = torch.randint(low=0, high=len(iterators), size=(1,))[0] # or any other dataset sampling logic...
            yield next(iterators[dataset_index])

class CSTRDataset(IterableDataset):
    '''
    This is the class with function of initialization, and of the iteration of the iterative call of the 
    different initialization for each batch used.
    '''
    def __init__(self,seq_len = 500, par_fixed = [8314, 801, 3137, 851, 1000, 4183, 0.004377, 350, 0.4],
             par_shifted = [69710000.0, -69710000.0, 20750000.0, 0.4, 350, 350], normalize = True,
                 shift_seed=None, input_seed=None, noise_seed=None): 
        '''
        par_fixed = [R,rho,cp,U,rhoJ,cJ,Fss,TRss,Cass]
        par_shifted = [E,lam,k0,Ca0,T0,TCin,UAJ]
        '''
        super(CSTRDataset).__init__()
        self.seq_len = seq_len+100
        self.par_fixed = par_fixed
        self.par_shifted = par_shifted
        self.normalize = normalize
        self.system_seed = shift_seed
        self.input_seed = input_seed
        self.noise_seed = noise_seed
        self.shift_rng = np.random.default_rng(shift_seed)  # source of randomness for system generation
        self.input_rng = np.random.default_rng(input_seed)  # source of randomness for input generation
        self.noise_rng = np.random.default_rng(noise_seed)  # source of randomness for noise generation

    
    def __iter__(self):
        
        n_skip = 100
        while True:
            t, u,y = CSTR_F.MC_parameters_CSTR(self.seq_len, self.par_fixed, self.par_shifted,\
                                               self.shift_rng,self.input_rng)

            u = u[n_skip:]
            y = y[n_skip:,:]

            if self.normalize:
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)
                u = (u - u.mean(axis=0)) / (u.std(axis=0) + 1e-6)



            e = self.noise_rng.normal(size=(self.seq_len-n_skip, 3)) * 0.1 # noise term, you should add a different one in each component
            y = y + e


            u = u.astype('float32')
            y = y.astype('float32')
            yield torch.tensor(y), torch.tensor(u)

class ConcatenatedIterableDataset(IterableDataset):
    def __init__(self, iterable_dataset1, iterable_dataset2):
        self.iterable_dataset1 = iterable_dataset1
        self.iterable_dataset2 = iterable_dataset2

    def __iter__(self):
        for data in self.iterable_dataset1:
            yield data
        for data in self.iterable_dataset2:
            yield data
 
                        
if __name__ == "__main__":



    # Create data loader
    mdlargs = {"strictly_proper":True, "mag_range": (0.8, 0.97), "phase_range": (0, math.pi / 2)}
    #train_ds = LinearDynamicalDataset(nx=5, nu=1, ny=1, seq_len=500, **mdlargs)

    train_ds = CSTRDataset(shift_seed=42, input_seed=445)
    # train_ds = LinearDynamicalDataset(nx=5, nu=2, ny=3, seq_len=1000)
    train_dl = DataLoader(train_ds, batch_size=2)
    batch_y, batch_u = next(iter(train_dl))
    print(batch_y.shape, batch_u.shape)