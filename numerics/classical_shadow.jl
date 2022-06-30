using LinearAlgebra
import Yao
using BitBasis
using ProgressBars
using Statistics
include("random_clifford.jl")


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
        b = comp_basis_measurement(U*rho*U')
        classical_snapshot = (2^n_qubits + 1)*U'*b*b'*U - I 
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