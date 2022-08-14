using LinearAlgebra
import Yao
using BitBasis
using ProgressBars
using StatsBase
include("../numerics/random_clifford.jl")
include("../numerics/utils.jl")


function rowsum!(h,i, tableau)
    T = tableau
    n = Int((size(tableau)[1]-1)/2) # tableau has (2n+1) \times (2n+1)
    function g(x_1, z_1, x_2, z_2)
        if x_1 == z_1 == 0
            return 0 
        elseif x_1 == z_1 == 1
            return z_2 - x_2 
        elseif (x_1 == 1) && (z_1 == 0)
            return z_2*(2*x_2 - 1)
        else
            return x_2*(1-2*z_2)
        end
    end

    rowsum! = 2*tableau[h, 2*n+1] + 2*tableau[i, 2*n+1]
    for j in 1:n
        rowsum! += g(T[i,j], T[i,j+n], T[h,j], T[h,j+n])
    end

    if mod(rowsum!, 0:3) == 0
        T[h, 2*n+1] = 0
    else
        T[h, 2*n+1] = 1
    end

    for j in 1:n
        T[h,j] = mod(T[h,j]+T[i,j], 0:1)
        T[h, j+n] = mod(T[h, j+n]+T[i,j+n], 0:1)
    end
end


""" state is given as a string resulting from the conversion of the decimal state number
to base=2, as in e.g. string(6, base=2, pad=4) where pad is equal to the number of qubits
in the system."""
function computational_basis_state_to_tableau(state)
    n = length(state)
    T = falses((2*n+1, 2*n+1)) # initialize tableau
    for i in 1:2*n
        T[i,i] = 1
    end

    for i in 1:n
        if string(state[i]) == "1"
            T[i+n, 2*n+1] = 1
        end
    end 

    return T 
end 


""" For computational basis measurements, only the diagonal elements are required, i.e. saving only 
the diagonal suffices. """
function measure_computational_basis_(diag_rho)
    n = Int(log2(length(diag_rho)))
    measurement_outcome = sample(pweights(real(diag_rho))) - 1 # comp. basis states start with 0
    return string(measurement_outcome, base=2, pad=n)
end


""" Computational basis measurement for a stabilizer state in tableau formalism. See https://arxiv.org/abs/quant-ph/0406196. """
function measure_computational_basis!(tableau)
    n = Int((size(tableau)[1]-1)/2)
    function measure_qubit(a, tableau)
        T = tableau
        p_exists = false
        p_idx = 0
        for i in 1:n
            if T[i+n, a] == 1
                p_exists = true
                p_idx = i+n
                break 
            end
        end

        if p_exists == true
            for i in 1:2*n
                if i == p_idx
                    continue 
                end
                if T[i,a] == 1
                    rowsum!(i,p_idx,T)
                end
            end
            T[p_idx-n, 1:end] = T[p_idx, 1:end]
            T[p_idx, 1:end] = falses(2*n+1)
            T[p_idx, 2*n+1] = rand([0,1])
            T[p_idx, a+n] = 1
            return T[p_idx, 2*n+1]

        else
            T[2*n+1, 1:end] *= 0
            for i in 1:n
                if T[i,a] == 1
                    rowsum!(2*n+1, i+n, T)
                end
            end
            return T[2*n+1, 2*n+1]
        end
    end

    measurement_outcome = ""
    for i_qubit in 1:n 
        i_measurement_outcome = measure_qubit(i_qubit, tableau)
        measurement_outcome *= string(Int(i_measurement_outcome))
    end
    return measurement_outcome
end


function H(a_qubit, tableau)
    n = Int((size(tableau)[1]-1)/2)
    tableau[1:end-1, 2*n+1] = vec_mod(tableau[1:end-1, 2*n+1] .+ tableau[1:end-1, a_qubit] .* tableau[1:end-1, a_qubit+n])
    h = copy(tableau[1:end-1, a_qubit])
    tableau[1:end-1, a_qubit] = tableau[1:end-1, a_qubit+n]
    tableau[1:end-1, a_qubit+n] = h
end


function S(a_qubit, tableau)
    n = Int((size(tableau)[1]-1)/2)
    tableau[1:end-1, 2*n+1] = vec_mod(tableau[1:end-1, 2*n+1] .+ tableau[1:end-1, a_qubit] .* tableau[1:end-1, a_qubit+n])
    tableau[1:end-1, a_qubit+n] = vec_mod(tableau[1:end-1, a_qubit+n] .+ tableau[1:end-1, a_qubit])
end


function CNOT(a_qubit, b_qubit, tableau)
    n = Int((size(tableau)[1]-1)/2)
    tableau[1:end-1, 2*n+1] = vec_mod(tableau[1:end-1, 2*n+1] .+ tableau[1:end-1, a_qubit] .* tableau[1:end-1, b_qubit+n] .* (tableau[1:end-1,b_qubit] .+ tableau[1:end-1, a_qubit+n] .+ 1))
    tableau[1:end-1, b_qubit] = vec_mod(tableau[1:end-1, b_qubit] .+ tableau[1:end-1, a_qubit])
    tableau[1:end-1, a_qubit+n] = vec_mod(tableau[1:end-1, b_qubit+n] .+ tableau[1:end-1, a_qubit+n])
end


""" Apply a unitary given as a quantum circuit (as the output of sample_random_Clifford()) to tableau. """
function apply_unitary(U, tableau)
    n = length(U)
    for i in 1:n
        unitary = U[i]
        if unitary[1] == "CX"
            CNOT(unitary[2], unitary[3], tableau)
        elseif unitary[1] == "H"
            H(unitary[2], tableau)
        elseif unitary[1] == "S"
            S(unitary[2], tableau)
        elseif unitary[1] == "X"
            H(unitary[2], tableau)
            S(unitary[2], tableau)
            S(unitary[2], tableau)
            H(unitary[2], tableau)
        elseif unitary[1] == "Y"
            S(unitary[2], tableau)
            H(unitary[2], tableau)
            S(unitary[2], tableau)
            S(unitary[2], tableau)
            H(unitary[2], tableau)
            S(unitary[2], tableau)
            S(unitary[2], tableau)
            S(unitary[2], tableau)
        elseif unitary[1] == "Z"
            S(unitary[2], tableau)
            S(unitary[2], tableau)
        end
    end
end

""" state is given as an explicit density matrix. """
function classical_snapshot(rho, n_qubit_clifford)
    n_qubits = Int(log2(size(rho)[1]))
    if n_qubit_clifford == true
        u = sample_random_Clifford(n_qubits)
        U = Clifford_circuit_to_matrix(u, n_qubits)
        b = measure_computational_basis(diag(U'*rho*U))
        b = computational_basis_state_to_tableau(b)
        apply_unitary(u, b)
        return b
    else
        # 1) construct unitary for rotation
        u = [] 
        for i_qubit in 1:n_qubits
            push!(u, sample_random_Clifford(1))
        end
        U = 1
        for i_qubit in 1:n_qubits
            U = LinearAlgebra.kron(U, Clifford_circuit_to_matrix(u[i_qubit], 1))
        end
        b = measure_computational_basis(diag(U'*rho*U))
        b = computational_basis_state_to_tableau(b)
        for i_qubit in 1:n_qubits
            apply_unitary(u[i_qubit], b)
        end
        return b
    end
end


""" Accumulate a shadow of given size (n_snapshots). Note that this is memory intensive, 
there might be scenarios where calculating on the fly might be better."""
function classical_shadow(rho, n_snapshots; n_qubit_clifford=false)
    n_qubits = Int(log2(size(rho)[1]))
    shadow = BitArray(undef, (n_snapshots, 2*n_qubits+1, 2*n_qubits+1))
    for i in 1:n_snapshots
        shadow[i, 1:end, 1:end] = BitArray(classical_snapshot(rho, n_qubit_clifford))
    end
    return shadow
end

""" If n_snapshots == 0, then all snapshots of the shadow are used. 
Note: measurement_primitive == "pauli" is still untested! """
function shadow_measurement(measurement_func, shadow; n_snapshots=0, measurement_primitive="clifford")
    n_qubits = Int((size(shadow)[2] -1)/2)
    if n_snapshots == 0
        n_snapshots = size(shadow)[1]
    else
        n_snapshots = min(n_snapshots, size(shadow)[1]) 
    end
    outcome = 0
    for i_snapshot in 1:n_snapshots
        outcome += measurement_func(shadow[i_snapshot, 1:end, 1:end])
    end
    outcome /= n_snapshots
    if measurement_primitive == "clifford"
        outcome *= (2^n_qubits +1)
    elseif measurement_primitive == "pauli"
        outcome *= 3^n_qubits
    end
    return outcome 
end

# for convenience a measurement function of the Pauli Z string 
function measurement_Z_n(tableau)
    T = copy(tableau)
    b = measure_computational_basis!(T)
    if isodd(count(==('1'), b))
        return -1
    else 
        return 1
    end
end

""" For pauli_string, input of the form "XYZIIZ" is required."""
function measurement_Pauli_string(tableau, pauli_string)
    T = copy(tableau)
    for (i, P) in enumerate(pauli_string)
        if P == 'X'
            H(i, T)
        elseif P == 'Y'
            H(i,T)
            S(i,T)
            H(i,T)
        end
    end
    b_ = measure_computational_basis!(T)
    b = ""
    for (i, P) in enumerate(pauli_string)
        if P == 'I'
            b *= '0'
        else
            b *= b_[i]
        end
    end
    if isodd(count(==('1'), b))
        return -1
    else 
        return 1
    end
end

#########################################
######## Outdated Implementation ########
#########################################
""" The following function are designed to operate on *explicitly written* density 
matrices and are thus terribly slow. We still keep this code as it provides a means
for testing new code.
------------
Actually, for small numbers of qubits this is even faster, still, the gold standard should be storing
results in tableau formalism (small memory footprint) and predict from there.   
"""



"""For now, let Yao do this. Maybe later try to beat their implementation."""
function comp_basis_measurement(rho, nshots=1; return_bitstring=false)
    measurement_outcome = measure(DensityMatrix(rho), nshots=nshots)
    if return_bitstring == true
        return measurement_outcome[1]
    else
        return onehot(measurement_outcome[1])
    end
end

function comp_one_qubit_basis(x)
    if x == 0
        return [1,0]
    else
        return [0,1]
    end
end


function construct_classical_snapshot(rho; measurement_primitive="single")
    n_qubits = Int(log2(size(rho)[1]))
    if measurement_primitive == "single"
        # 1) construct unitary for rotation
        unitary = [] 
        for i_qubit in 1:n_qubits
            push!(unitary, Clifford_circuit_to_matrix(sample_random_Clifford(1),1))
        end
        U = 1
        for i_qubit in 1:n_qubits
            U = LinearAlgebra.kron(U, unitary[i_qubit])
        end
        b = comp_basis_measurement(U*rho*U', return_bitstring=true) 
        classical_snapshot = 1
        for i_qubit in 1:n_qubits
            b_ = comp_one_qubit_basis(b[i_qubit])
            classical_snapshot = LinearAlgebra.kron(classical_snapshot, 3*unitary[i_qubit]'*b_*b_'*unitary[i_qubit] - I)
        end
        return classical_snapshot
    else
        U = Clifford_circuit_to_matrix(sample_random_Clifford(n_qubits), n_qubits)
        b = comp_basis_measurement(U'*rho*U)
        #classical_snapshot = (2^n_qubits + 1)*U*b*b'*U' - I 
        classical_snapshot = U*b*b'*U' 
        return classical_snapshot
    end
end
            



function construct_classical_shadow(rho; n_snapshots=1, measurement_primitive="single")
    classical_shadow = Array{Matrix}(undef, n_snapshots)
    Threads.@threads for i_snapshot in ProgressBar(1:n_snapshots)
        classical_shadow[i_snapshot] = construct_classical_snapshot(rho, measurement_primitive=measurement_primitive)
    end
    return classical_shadow
end


function shadow_prediction(measurement_operator, classical_shadow; k_sample_means=1)
    k_means = Array{Float64}(undef, k_sample_means)
    n_snapshots = size(classical_shadow)[1]
    k_snapshots = Int(n_snapshots/k_sample_means)
    for j in 1:k_sample_means
        j_mean = 0
        for i_snapshot in 1+(j-1)*k_snapshots: j*k_snapshots
            j_mean += tr(measurement_operator*classical_shadow[i_snapshot])
        end
        j_mean /= k_snapshots
        k_means[j] = real(j_mean)
    end
    return median(k_means)
end