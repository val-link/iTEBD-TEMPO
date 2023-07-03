# Implementation of TEMPO
# Author: Valentin Link (valentin.link@tu-dresden.de)

import numpy as np
from scipy.integrate import dblquad
from scipy.linalg import expm, qr
from typing import Callable, Optional
from tqdm import tqdm
from copy import deepcopy
from mpnum import MPArray
from mpnum import dot as mpdot


class BathCorrelation():
    """ A class to store the bath correlation function. """

    def __init__(self, bcf: Callable[[float], complex]):
        """
        :param bcf: The bath correlation function.
        """
        self.bcf = bcf

    def compute_eta(self, n: int, delta: float) -> np.ndarray:
        """
        Computes the discretized bath correlation function (eta) for n time steps delta.
        :param n: Number of time steps.
        :param delta: Time step.
        :return: Discretized bath correlation function.
        """

        eta = np.zeros(n, dtype=np.complex128)
        eta[0] += dblquad(lambda s, t: np.real(self.bcf(t - s)), 0, delta, lambda t: 0, lambda t: t)[0]
        eta[0] += dblquad(lambda s, t: np.imag(self.bcf(t - s)), 0, delta, lambda t: 0, lambda t: t)[0] * 1j

        for k in range(1, n):
            eta[k] += dblquad(lambda s, t: np.real(self.bcf(t - s)), k * delta, (k + 1) * delta, 0, delta)[0]
            eta[k] += dblquad(lambda s, t: np.imag(self.bcf(t - s)), k * delta, (k + 1) * delta, 0, delta)[0] * 1j

        return eta


class TEMPO():
    """ A class to compute and approximate the influence functional unsing iTEMPO and compute dynamics. """

    def __init__(self, l_vals: np.ndarray, delta: float, bcf: Callable[[float], complex], n: int, n_c: Optional[int] = 0):
        """
        :param l_vals: Real eigenvalues of the system-bath coupling operator.
        :param n: Number of time steps.
        :param delta: Time step for Trotter splitting. Total evolution time is T=n*delta.
        :param bcf: Bath correlation function.
        :param n_c: Memory cutoff. The effort for computations is O(n * n_c). n_c must be chosen large enough, otherwise bond dimension of the influence functional MPO will become large. n_c = 0 amounts to no cutoff.
        """
        if n_c == 0 or n_c > n:
            self.n_c = n
        else:
            self.n_c = n_c
        self.l_vals = l_vals  # add trivial dimension to recover f at previous times
        self.l_dim = self.l_vals.size
        self.nu_dim = self.l_vals.size ** 2 + 1
        self.n = n
        self.bcf = BathCorrelation(bcf)
        self.delta = delta
        self.eta = self.bcf.compute_eta(self.n_c, delta)
        self.i_tens = np.empty((self.n_c + 1, self.nu_dim, self.nu_dim), dtype=np.complex128)
        self.l_diff = np.empty((self.nu_dim - 1), dtype=np.complex128)
        self.l_sum = np.empty((self.nu_dim - 1), dtype=np.complex128)
        for nu in range(self.nu_dim - 1):
            i, j = int(nu / self.l_dim), nu % self.l_dim
            self.l_diff[nu] = self.l_vals[i] - self.l_vals[j]
            self.l_sum[nu] = self.l_vals[i] + self.l_vals[j]
        self.l_diff = np.pad(self.l_diff, [(0, 1)])
        self.l_sum = np.pad(self.l_sum, [(0, 1)])

        for k in range(self.n_c):
            self.i_tens[k, :, :] = np.exp(-self.eta[k].real * np.outer(self.l_diff, self.l_diff) - 1j * self.eta[k].imag * np.outer(self.l_sum, self.l_diff))
        self.kron_delta = np.identity(self.nu_dim)
        self.f = None
        self.f_caps = None

    def compute_f(self, rtol: Optional[float] = 1e-8, rank: Optional[int] = None, contraction: Optional[str] = 'Jorgensen'):
        """
        Compute the influence functional F in mps form by using SVD compression to keep bond dimensions manageable.

        :param rtol: Relative error for SVD compression.
        :param rank: Maximum allowed rank in SVD compression (None = unrestricted).
        :param contraction: Different contraction schemes are available. 'Jorgensen' is fast but bonds are larger. 'Strathearn' is slow but results in the lowest bond dimensions. 'diagonal' is experimental.
        """

        if contraction == 'Strathearn':
            self._compute_f_Strathearn(rtol, rank)
        elif contraction == 'Jorgensen':
            self._compute_f_Jorgensen(rtol, rank)
        elif contraction == 'diagonal':
            self._compute_f_diagonal(rtol, rank)
        else:
            raise ValueError('unknown contraction scheme, available schemes are Jorgensen, Strathearn, diagonal')
        print('ranks: ', self.f.ranks)
        self.f._lt._ltens = [np.einsum('abc->cba', f) for f in self.f._lt._ltens]
        self.f._lt._ltens.reverse()
        # compute cap tensors to get f at previous time-steps
        x = np.ones((1, 1))
        self.f_caps = []
        self.f_caps.append(np.ones((1, 1), dtype=np.complex128))
        for k in range(1, self.n):
            self.f_caps.append(self.f._lt._ltens[-k][:, -1, :].dot(self.f_caps[-1]))
        self.f_caps.reverse()
        self.f_phys = MPArray([t[:, :-1, :] for t in self.f._lt._ltens])

    def _compute_f_Strathearn(self, relerr: float, rank: int):
        """
        Represent the influence functional F in tensor-train form while using SVD compression to keep bond dimensions manageable.
        Strathearn contraction scheme
        """

        a0_tensor = np.einsum('a,ij,j,b->bija', np.ones((1)), self.kron_delta, np.diagonal(self.i_tens[0]), np.ones((1)))  # modified a tensor for first time step
        a_tensor = np.einsum('a,ij,jb,j->bija', np.ones((1)), self.kron_delta, self.kron_delta, np.diagonal(self.i_tens[0]))
        b_tensor = np.einsum('ij,ab,kaj->kbija', self.kron_delta, self.kron_delta, self.i_tens[:self.n_c, :, :], optimize='greedy')
        c_tensor = np.einsum('b,ij,kaj->kbija', np.ones((1)), self.kron_delta, self.i_tens[:, :, :], optimize='greedy')
        self.f = MPArray([a0_tensor])
        for k in tqdm(range(1, self.n), desc='building influence functional'):
            if k <= self.n_c:
                B = [c_tensor[k]]
                for m in range(k - 1):
                    B.append(b_tensor[k - 1 - m])
                B.append(a_tensor)
                B1 = deepcopy(self.f._lt._ltens)  # lower B tensor
                B1.append(self.kron_delta.reshape((1, self.nu_dim, self.nu_dim, 1)))  # attach identity to lower B tensor to get proper shape for mpdot
                B1_mpnum = MPArray(B1)
                B2_mpnum = MPArray(B)
                B.clear()
                self.f = mpdot(B1_mpnum, B2_mpnum, axes=(1, 1))
                self.f.compress(relerr=relerr, rank=rank, direction='left')
                self.f.compress(relerr=relerr, rank=rank, direction='right')
            else:
                B1 = deepcopy(self.f._lt._ltens)  # lower B tensor
                B1_a = B1[:k - self.n_c]
                B1_b = B1[k - self.n_c:]
                B1_b.append(self.kron_delta.reshape((1, self.nu_dim, self.nu_dim, 1)))  # attach identity to lower B tensor to get proper shape for mpdot
                B1_mpnum = MPArray(B1_b)
                self.f = mpdot(B1_mpnum, B2_mpnum, axes=(1, 1))
                self.f.compress(relerr=relerr, rank=rank, direction='left')
                self.f.compress(relerr=relerr, rank=rank, direction='right')
                self.f._lt._ltens = B1_a + self.f._lt._ltens

        self.f = self.f.sum(axes=1)

    def _compute_f_Jorgensen(self, relerr: float, rank: int):
        """
        Represent the influence functional F in tensor-train form while using SVD compression to keep bond dimensions manageable.
        Jorgensen contraction scheme
        """

        a0_tensor = np.einsum('a,ij,j,b->aibj', np.ones((1)), self.kron_delta, np.diagonal(self.i_tens[0]), np.ones((1)))  # modified a tensor for first time step
        a_tensor = np.einsum('a,ij,jb,j->aibj', np.ones((1)), self.kron_delta, self.kron_delta, np.diagonal(self.i_tens[0]))
        b_tensor = np.einsum('ij,ab,kaj->kiabj', self.kron_delta, self.kron_delta, self.i_tens[:self.n_c, :, :], optimize='greedy')
        c_tensor = np.einsum('b,ij,kaj->kiabj', np.ones((1)), self.kron_delta, self.i_tens[:, :, :], optimize='greedy')
        c_end = np.sum(c_tensor[self.n_c - 1], axis=3).reshape([self.nu_dim, self.nu_dim, 1, 1])
        self.f = MPArray([a0_tensor] + [c_tensor[k] for k in range(1, self.n_c - 1)] + [c_end])
        f_tens = [a0_tensor]
        D = [a_tensor] + [b_tensor[m] for m in range(1, self.n_c - 1)] + [c_end]
        for k in tqdm(range(1, self.n), desc='building influence functional'):
            if k <= self.n - self.n_c:
                D1 = deepcopy(self.f._lt._ltens[1:])  # upper D tensor
                D1.append(np.ones((1, 1, 1, 1)))
                D1_mpnum = MPArray(D1)
                D2_mpnum = MPArray(D)
                D1.clear()
            else:
                if k == self.n - 1:
                    a_end = np.sum(a_tensor, axis=3).reshape([1, self.nu_dim, self.nu_dim, 1])
                    D = [a_end]
                else:
                    b_end = np.sum(b_tensor[self.n - k - 1], axis=3).reshape([self.nu_dim, self.nu_dim, self.nu_dim, 1])
                    D = [a_tensor] + [b_tensor[m] for m in range(1, self.n - k - 1)] + [b_end]
                D1 = deepcopy(self.f._lt._ltens[1:])  # upper D tensor
                D1_mpnum = MPArray(D1)
                D2_mpnum = MPArray(D)
                D1.clear()
                D.clear()

            self.f = mpdot(D2_mpnum, D1_mpnum, axes=(1, 0))
            self.f.compress(relerr=relerr, rank=rank, direction='left')
            self.f.compress(relerr=relerr, rank=rank, direction='right')
            f_tens.append(self.f._lt._ltens[0])
        self.f = MPArray(f_tens).ravel()

    def _compute_f_diagonal(self, relerr: float, rank: int):
        """
        Represent the influence functional F in tensor-train form while using SVD compression to keep bond dimensions manageable.
        Diagonal contraction scheme
        """
        a_tensor = np.einsum('a,ij,jb,j->bija', np.ones((1)), self.kron_delta, self.kron_delta, np.diagonal(self.i_tens[0]), optimize='greedy')
        b_tensor = np.einsum('ij,ab,kaj->kbija', self.kron_delta, self.kron_delta, self.i_tens[:self.n_c, :, :], optimize='greedy')

        def split_tensor(x):
            """ split a tensor with four legs into two tensors with three legs """
            shape = x.shape
            q, r = qr(x.reshape([shape[0] * shape[1], shape[2] * shape[3]]))
            return q.reshape([1, shape[0], shape[1], q.shape[1]]), r.reshape([r.shape[0], shape[2], shape[3], 1])

        b1, b2 = split_tensor(b_tensor[-1])

        f_tens = [b1.sum(axis=1), b2.sum(axis=1)] * (self.n - self.n_c + 1)

        for k in tqdm(range(2, self.n_c + 1), desc='building influence functional'):
            if k < self.n_c:
                b1, b2 = split_tensor(b_tensor[-k])
                f2 = [b1, b2] * (self.n - self.n_c + k)
                f2[0] = f2[0].sum(axis=1)
                f2[-1] = f2[-1].sum(axis=1)
            else:
                a1, a2 = split_tensor(a_tensor)
                f2 = [a1, a2] * self.n
                f2[0] = f2[0].sum(axis=1)
                f2[-1] = f2[-1].sum(axis=1)
            self.f = mpdot(MPArray(f2[1:-1]), MPArray(f_tens), axes=(0, -1))
            self.f._lt._ltens = [f2[0]] + self.f._lt._ltens + [f2[-1]]
            self.f.compress(relerr=relerr, rank=rank, direction='left')
            f_tens = deepcopy(self.f._lt._ltens)

        self.f = MPArray(f_tens).group_sites(2).ravel()

    def f_m(self, m: int) -> MPArray:
        """ Return the influence functional for m < n time steps. """
        if self.f == None:
            raise IndexError('the influence functional has not yet been computed, run self.compute_f first')
        return MPArray(self.f._lt._ltens[:m - 1] + [self.f._lt._ltens[m - 1].dot(self.f_caps[m - 1])])

    def get(self, i_path: np.ndarray) -> complex:
        """ Computes the influence of a single path with indices i_path. """
        if self.f == None:
            raise IndexError('the influence functional has not yet been computed, run self.compute_f first')
        if i_path.size != self.n or type(i_path[0]) is not np.int32:
            raise TypeError('i_path must be an integer numpy array of size self.n')
        return self.f.get(i_path).to_array()

    def get_exact(self, i_path: np.ndarray) -> complex:
        """ Computes the exact influence of a single path with indices i_path of l_vals. """
        if self.eta.size < self.n:
            self.eta = self.bcf.compute_eta(self.n, self.delta)
        if i_path.size != self.n or type(i_path[0]) is not np.int32:
            raise TypeError('i_path must be an integer numpy array of size self.n')
        l_diff_path = [self.l_diff[i] for i in np.flip(i_path)]
        l_sum_path = [self.l_sum[i] for i in np.flip(i_path)]
        are = np.concatenate((np.flip(self.eta[:self.n].real), np.zeros(self.n)))
        aim = np.concatenate((np.flip(self.eta[:self.n].imag), np.zeros(self.n)))
        return np.exp(-np.dot(l_diff_path, np.convolve(are, l_diff_path, 'valid')[:self.n]) - 1j * np.dot(l_diff_path, np.convolve(aim, l_sum_path, 'valid')[:self.n]))

    def evolve(self, h_s: np.ndarray, rho_0: np.ndarray) -> np.ndarray:
        """
        Compute the time evolution for n time steps.

        :param h_s: System Hamiltonian in the eigenbasis of the coupling operator.
        :param rho_0: System initial state.
        :param n: Number of time-steps for the propagation.
        :return: Time evolution of density matrix.
        """
        assert self.f is not None, "the influence functional has not yet been computed, run self.compute_f first"
        liu_s_half = np.kron(expm(-1j * h_s * self.delta / 2), expm(1j * h_s * self.delta / 2).T)
        liu_s = liu_s_half @ liu_s_half

        nu_dim = self.l_dim ** 2
        u = np.einsum('ab,bc->abc', np.identity(nu_dim), liu_s.T)
        u_half = np.einsum('ab,bc->abc', np.identity(nu_dim), liu_s_half.T)

        rho_t = np.empty((self.n + 1, self.l_dim, self.l_dim), dtype=np.complex128)
        rho_t[0] = rho_0

        tens = np.einsum('b,c->bc', np.ones((1)), liu_s_half @ rho_0.flatten())

        for i in tqdm(range(self.n), desc='time evolution running'):
            evol_tens_half = np.einsum('abc,dbe->adce', self.f._lt._ltens[i][:, :-1, :], u_half, optimize='greedy')
            tens_ = np.einsum('bc,bcde->de', tens, evol_tens_half, optimize='greedy')
            rho_t[i + 1] = np.einsum('ab,a->b', tens_, self.f_caps[i].flatten(), optimize='greedy').reshape(rho_0.shape)
            evol_tens = np.einsum('abc,dbe->adce', self.f._lt._ltens[i][:, :-1, :], u, optimize='greedy')
            tens = np.einsum('bc,bcde->de', tens, evol_tens, optimize='greedy')
        return rho_t
