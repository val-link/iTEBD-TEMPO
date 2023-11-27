# Implementation of iTEBD-TEMPO
# Author: Valentin Link (valentin.link@tu-dresden.de)

using LinearAlgebra
using TensorOperations
using OMEinsum
using ProgressMeter
import Arpack.eigs
import Cubature.hcubature_v


"""
    compute_η(bcf::Function, Δ::Float64, n::Int64) -> Vector{ComplexF64}

Compute the discretized bath correlation function for the complex function `bcf` and `n` time steps `Δ`.
"""
function compute_η(bcf::Function, Δ::Float64, n::Int64)
    function bcf_re(t, v)
        @. v = real(bcf(t[1, :] - t[2, :]))
    end

    function bcf_im(t, v)
        @. v = imag(bcf(t[1, :] - t[2, :]))
    end

    function bcf_mod_im(t, v)
        @. v = imag((t[1, :] - t[2, :]) |> t -> t > 0 ? bcf(t) : -bcf(t))
    end

    eta = zeros(ComplexF64, n + 1)
    for i in 1:n+1
        eta[i] = (hcubature_v(bcf_re, [(i - 1) * Δ, 0], [(i) * Δ, Δ], abstol=1e-7))[1]
        # boundary term needs special care
        if i == 1
            eta[1] += im * (hcubature_v(bcf_mod_im, [0, 0], [Δ, Δ], abstol=1e-7))[1]
            eta[1] /= 2
        else
            eta[i] += im * (hcubature_v(bcf_im, [(i - 1) * Δ, 0], [(i) * Δ, Δ], abstol=1e-7))[1]
        end
    end
    return eta
end


"""
    iTEBD_TEMPO(s_vals::Vector{Float64}, Δ::Float64, bcf::Function, n_c::Int)

`iTEBD_TEMPO` allows to compute open system dynamics using an iTEBD based approach. The constructor does not perform the algorithm itself, but initializes the `iTEBD_TEMPO` struct with a trivial influence functional. Use the `compute_f!` function to start the contraction algorithm. 


    (itt::iTEBD_TEMPO)(ν_path::Vector{Int64})

When called as a function an `iTEBD_TEMPO` struct returns the value of the tensor for a given path `ν_path` (vector of indices). Both the exact and approximated values are computed. This can be used to check the accuracy of the compressed influence functional.

# Example

Construct a new `iTEBD_TEMPO` struct and start iTEBD-TEMPO alogorithm.
```
    itt = iTEBD_TEMPO(s_vals, Δ, bcf, n_c)
    compute_f!(itt, 1e-9)
``` 

Compute the influence of a random path.
```
    import StatsBase.sample
    influence, influence_exact = itt(sample(collect(1:itt.s_dim^2)), 100)
```

# Related functions

 -  To compute time evolution use `evolve`.
 -  To compute steady states use `steadystate`. 
"""
mutable struct iTEBD_TEMPO
    n_c::Int64
    n_c_eff::Int64
    s_dim::Int64
    ν_dim::Int64
    η::Vector{ComplexF64}
    Δ::Float64
    s_diff::Vector{Float64}
    s_sum::Vector{Float64}
    f::Array{ComplexF64,3}
    v_r::Array{ComplexF64,1}
    v_l::Array{ComplexF64,1}
end
function iTEBD_TEMPO(s_vals::Vector{Float64}, Δ::Float64, bcf::Function, n_c::Int)
    n_c_eff = n_c
    s_dim = length(s_vals)
    ν_dim = s_dim^2 + 1
    η = compute_η(bcf, Δ, n_c)
    s_diff = zeros(Float64, ν_dim)
    s_sum = zeros(Float64, ν_dim)
    for ν in 1:(ν_dim-1)
        i, j = 1 + Int(floor((ν - 1) / s_dim)), 1 + (ν - 1) % s_dim
        s_diff[ν] = s_vals[j] - s_vals[i]
        s_sum[ν] = s_vals[j] + s_vals[i]
    end
    f = ones(ComplexF64, 1, ν_dim, 1)
    v_r = Vector(ones(ComplexF64, 1))
    v_l = Vector(ones(ComplexF64, 1))
    return iTEBD_TEMPO(n_c, n_c_eff, s_dim, ν_dim, η, Δ, s_diff, s_sum, f, v_r, v_l)
end
function (itt::iTEBD_TEMPO)(ν_path::Vector{Int64})
    n = length(ν_path)
    n > length(itt.η) && error("length of the path must not be longer than n_c")
    # this computes the exact value
    s_diff_path = [itt.s_diff[ν_path[i]] for i in eachindex(ν_path)]
    s_sum_path = [itt.s_sum[ν_path[i]] for i in eachindex(ν_path)]
    exponent = 0.0 + 0 * im
    for s in 1:n, v in 1:s
        exponent += -s_diff_path[s] * real(itt.η[1+s-v]) * s_diff_path[v] - im * s_diff_path[s] * imag(itt.η[1+s-v]) * s_sum_path[v]
    end
    # this is the value computed with the compressed influence functional
    val = conj(itt.v_l)'
    for t in eachindex(ν_path)
        val = val * itt.f[:, ν_path[t], :]
    end
    return val * itt.v_r, exp(exponent)
end


"""
    iTEBD_apply_gate(gate::Array{ComplexF64, 4}, A::Array{ComplexF64, 3}, sAB::Vector{ComplexF64}, B::Array{ComplexF64, 3}, sBA::Vector{ComplexF64}, rtol::Float64; ctol::Float64=1e-13)

single iTEBD step, scheme adapted from https://www.tensors.net/mps. rtol is the relative SVD compression tolerance. 
"""
function iTEBD_apply_gate(gate::Array{ComplexF64,4}, A::Array{ComplexF64,3}, sAB::Vector{ComplexF64}, B::Array{ComplexF64,3}, sBA::Vector{ComplexF64}, rtol::Float64; ctol::Float64=1e-13)
    # renormalize weights
    sAB = sAB * norm(sBA)
    sBA = sBA / norm(sBA)

    # ensure weights are above tolerance (needed for inversion)
    @. sBA[abs(sBA)<ctol] = ctol

    # MPS - gate contraction
    d1 = size(gate)[2]
    d2 = size(gate)[end]
    rank_BA = size(sBA)[1]

    @tensor C[-1, -2, -3, -4] := diagm(sBA)[-1, 1] * A[1, 5, 2] * diagm(sAB)[2, 4] * B[4, 6, 3] * diagm(sBA)[3, -4] * gate[5, -2, 6, -3]
    u, s_vals, v = svd(reshape(C, d1 * rank_BA, d2 * rank_BA), full=false)
    v = conj(transpose(v))

    # find rank truncation
    s_vals_sum = cumsum(s_vals) ./ sum(s_vals)
    rank_rtol = searchsortedfirst(s_vals_sum, 1 - rtol)
    rank_new = minimum([length(s_vals), rank_rtol])
    u = reshape(u[:, 1:rank_new], size(sBA)[1], d1 * rank_new)
    v = reshape(v[1:rank_new, :], rank_new * d2, rank_BA)

    # factor out sAB weights from A and B
    A = reshape(diagm(1 ./ sBA) * u, size(sBA)[1], d1, rank_new)
    B = reshape(v * diagm(1 ./ sBA), rank_new, d2, rank_BA)

    # new weights
    sAB = s_vals[1:rank_new]

    return A, Vector{ComplexF64}(sAB), B, Vector{ComplexF64}(sBA)
end


"""
    compute_f!(itt::iTEBD_TEMPO, rtol::Float64)

Compute the infinite influence functional tensor f for an `iTEBD_TEMPO` problem using the iTEBD-TEMPO algorithm. `rtol` is the relative SVD compression tolerance. 
"""
function compute_f!(itt::iTEBD_TEMPO, rtol::Float64)
    A = ones(ComplexF64, 1, itt.ν_dim, 1)
    B = ones(ComplexF64, 1, itt.ν_dim, 1)
    sAB = Vector(ones(ComplexF64, 1))
    sBA = Vector(ones(ComplexF64, 1))
    rank_is_one = true
    δ = diagm(ones(itt.ν_dim))
    @showprogress showspeed = true desc = "building influence functional..." for k in 0:(itt.n_c-1)
        i_tens = exp.(-real(itt.η[itt.n_c-k]) * (itt.s_diff * itt.s_diff') - im * imag(itt.η[itt.n_c-k]) * (itt.s_sum * itt.s_diff'))

        if k == itt.n_c - 1
            One = Vector(ones(ComplexF64, 1))
            @ein gate[j, a, b, i] := One[a] * δ[i, j] * δ[j, b] * diag(i_tens)[j]
        else
            @ein gate[j, a, b, i] := δ[i, j] * δ[a, b] * i_tens[a, j]
        end

        if k % 2 == 0
            B, sBA, A, sAB = iTEBD_apply_gate(gate, B, sBA, A, sAB, rtol)
        else
            A, sAB, B, sBA = iTEBD_apply_gate(gate, A, sAB, B, sBA, rtol)
        end

        if rank_is_one
            if all([length(sAB) == 1, length(sBA) == 1])
                # reset to initial mps if rank is still one
                A = ones(ComplexF64, 1, itt.ν_dim, 1)
                B = ones(ComplexF64, 1, itt.ν_dim, 1)
                sAB = Vector(ones(ComplexF64, 1))
                sBA = Vector(ones(ComplexF64, 1))
            else
                rank_is_one = false
                itt.n_c_eff = itt.n_c - k + 1
                k == 0 && (@warn "The memory cutoff n_c may be too small for the given rtol value. The algorithm may become unstable and inaccurate. It is recommended to increase n_c until this message does no longer appear.")
            end
        end
    end
    sAB = diagm(sAB)
    sBA = diagm(sBA)

    if (size(sAB)[1] == 1)
        itt.f = ones(ComplexF64, 1, itt.ν_dim, 1)
        itt.v_r = Vector(ones(ComplexF64, 1))
        itt.v_l = Vector(ones(ComplexF64, 1))
        println("rank 1 (trivial influence functional)")
        return
    end

    @tensor f[-1, -2, -3, -4] := sAB[-1, 1] * B[1, -2, 2] * sBA[2, 3] * A[3, -3, -4]
    itt.f = (dropdims(f, dims=tuple(findall(size(f) .== 1)...)))

    # compute f[:,-1,:]^\inf = v_r * v_l^T using Lanczos
    w, v_r = eigs(itt.f[:, end, :], nev=1, which=:LR)
    w, v_l = eigs(transpose(itt.f[:, end, :]), nev=1, which=:LR)
    itt.v_r = v_r[:]
    itt.v_l = v_l[:] / (conj(v_l[:])' * v_r[:])
    println("rank ", size(f)[1])
end


"""
    evolve(itt::iTEBD_TEMPO, h_s::Union{Array{ComplexF64,2}, Function}, ρ_0::Array{ComplexF64,2}, n::Int64) -> Array{ComplexF64,3}

Compute the time evolution for `n` time steps with initial state `ρ_0` (density matrix). `h_s` can be a static Hamitlonian or function in case of a time-dependent Hamiltonian. Note that in the case of a time-dependent Hamiltonian the time dependence is discretized as well.


# Example

The function simply returns the density matrix at each time step as a 3 dimensional array, where the first axis is time. 
```
    ρ_t = evolve(itt, h_s, ρ_0, n)
    t_eval = collect(0:n) * itt.Δ
    plot(t_eval, real.(ρ_t[:,1,1])) # plot the dynamics of <1|ρₜ|1>
```
"""
function evolve(itt::iTEBD_TEMPO, h_s::Array{ComplexF64,2}, ρ_0::Array{ComplexF64,2}, n::Int64)
    @assert size(h_s) == (itt.s_dim, itt.s_dim) "Hamiltonian has invalid shape."
    @assert size(ρ_0) == (itt.s_dim, itt.s_dim) "Initial state has invalid shape."

    liu_s = transpose(kron(transpose(exp(im * h_s * itt.Δ / 2)), (exp(-im * h_s * itt.Δ / 2))))
    @ein u[a, b, c] := liu_s[a, b] * liu_s[b, c]
    ρ_t = zeros(ComplexF64, n + 1, itt.s_dim, itt.s_dim)
    ρ_t[1, :, :] = ρ_0
    f = itt.f[:, 1:end-1, :]
    @tensor evol_tens[-1, -2, -3, -4] := f[-1, 1, -3] * u[-2, 1, -4]
    @ein state[a, b] := itt.v_l[a] * ρ_0[:][b]

    @showprogress showspeed = true desc = "performing time evolution..." for i in 1:n
        @tensor st[-1, -2] := state[1, 2] * evol_tens[1, 2, -1, -2]
        state = st
        @tensor ρ[-1] := itt.v_r[1] * state[1, -1]
        ρ_t[i+1, :, :] = reshape(ρ, size(ρ_0))
    end
    return ρ_t
end
function evolve(itt::iTEBD_TEMPO, h_s::Function, ρ_0::Array{ComplexF64,2}, n::Int64)
    h_0 = h_s(0)
    @assert typeof(h_0) == Matrix{ComplexF64} "Function h_s must return a complex matrix."
    @assert size(h_0) == (itt.s_dim, itt.s_dim) "Hamiltonian has invalid shape."
    @assert size(ρ_0) == (itt.s_dim, itt.s_dim) "Initial state has invalid shape."

    ρ_t = zeros(ComplexF64, n + 1, itt.s_dim, itt.s_dim)
    ρ_t[1, :, :] = ρ_0
    f = itt.f[:, 1:end-1, :]
    @ein state[a, b] := itt.v_l[a] * ρ_0[:][b]
    δ = diagm(ones(itt.s_dim^2))
    @showprogress showspeed = true desc = "performing time evolution..." for i in 1:n
        u_s = exp(-im * h_s(itt.Δ * (i - 1)) * itt.Δ / 2)
        liu_s = transpose(kron(transpose(u_s'), u_s))
        @ein u[a, b, c] := liu_s[a, b] * liu_s[b, c]
        @tensor st[-1, -2] := state[1, 2] * f[1, 3, -1] * u[2, 3, -2]
        state = st
        @tensor ρ[-1] := itt.v_r[1] * state[1, -1]
        ρ_t[i+1, :, :] = reshape(ρ, size(ρ_0))
    end
    return ρ_t
end


"""
    steadystate(itt::iTEBD_TEMPO,h_s::Array{ComplexF64,2}) -> Array{ComplexF64,2}

Compute the steady state for Hamiltonian `h_s` using Lanczos.
"""
function steadystate(itt::iTEBD_TEMPO, h_s::Array{ComplexF64,2})
    @assert size(h_s) == (itt.s_dim, itt.s_dim) "Hamiltonian has invalid shape."

    liu_s = transpose(kron(transpose(exp(im * h_s * itt.Δ / 2)), (exp(-im * h_s * itt.Δ / 2))))
    @ein u[a, b, c] := liu_s[a, b] * liu_s[b, c]

    f = itt.f[:, 1:end-1, :]
    @tensor evol_tens[-1, -2, -3, -4] := f[-1, 1, -3] * u[-2, 1, -4]

    w, v = eigs(transpose(reshape(evol_tens, prod(size(evol_tens)[1:2]), prod(size(evol_tens)[3:4]))), nev=1, which=:LR)

    ρ_ss = reshape(conj(itt.v_r)' * reshape(v, length(itt.v_r), itt.s_dim^2), itt.s_dim, itt.s_dim)
    return ρ_ss / tr(ρ_ss)
end