# iTEBD-TEMPO
An implementation of the open quantum system algorithm introduced in Ref. https://arxiv.org/abs/2307.01802. It uses a representation of <strong>open quantum system dynamics</strong> in terms of infinite tensor networks. Specifically, the time-discrete influence functional is generated in a periodic tensor train form using infinite time evolving block decimation (iTEBD). This object can then be used to compute dynamics and steady states for open quantum systems <strong>with arbitrary stationary Gaussian baths </strong> (arbitrary stationary bath correlation function). Both Julia and Python implementations are available.

# Examples

I do not provide a full documentation. The code is kept very simple, please read through the source file and the paper for details.

## Basic Usage
Currently, the python implementation can be used to solve open system problems with Hamiltonian
$$H(t)=H_S\otimes 1_E+S\otimes B(t)$$
where $H_S$ is the system Hamiltonian, $S$ is a diagonal Hermitian coupling operator and $B(t)$ characterizes the Gaussian quantum bath with correlation function
$$\text{BCF}(t-s)=\text{tr}_E[ \rho_E(0)B(t)B(s)] .$$
The following inputs are required. The bath correlation function $\text{BCF}(t)$, the time step $\Delta$ used for the Trotter splitting, the bath memory time step $N_c$ (must be chosen large enough) and the eigenvalues (diagonal elements) of $S$.

    >>> MyiTEBD_TEMPO = iTEBD_TEMPO(np.diagonal(S), Delta, BCF, N_c)

To computing an open system problem, two steps are required. First the influence functional has to be contracted to tensor train form using iTEBD with SVD compression at a given tolerance. Decreasing the tolerance will yield a more accurate representation. This is the computationally demanding step.

    >>> MyiTEBD_TEMPO.compute_f(rtol=1e-6)

Once the influence functional has been computed, we can compute dynamics. This returns the time evolution of the reduced system state $\rho(t)$ for $N$ time steps $\Delta$.

    >>> Rho_t = MyiTEBD_TEMPO.evolve(H_S, Rho_0, N)

The computational effort is just linear in $N$. Time-dependent system Hamiltonians are not yet supported. The steady state can be computed directly using Lanczos eigenvalue solver.

    >>> Rho_ss = MyiTEBD_TEMPO.steadystate(H_S)
All results should be checked for convergence with respect to the SVD compressions tolerance (rtol) and the Trotterization time step $\Delta$ (second order Trotter splitting is implemented which yields a $O(\Delta^3)$ error). 

Note that time-dependent Hamiltonians are supported in the Julia version only.

## Example Notebook
For a full example and test, consider cheking out the example notebook provided in the repo.



# Citing
Please cite the corresponding preprint https://arxiv.org/abs/2307.01802.
