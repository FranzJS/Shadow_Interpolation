using Random
using LinearAlgebra

function random_Pauli_measurement_procedure(n_measurements, system_size)
    measurements = Array{String, 2}(undef, n_measurements, system_size)
    rand!(measurements, ["X", "Y", "Z"])
    return measurements
end


# ToDo reimplement this for testing
function derandomized_Pauli_measurement_procedure(n_measurements::AbstractFloat, observables::Matrix{String}, precision::AbstractFloat)
    """This function is implemented according to https://arxiv.org/abs/2103.07510, the implementation 
    is best understood with respect to Eq. (B11). Indices were kept as given in the paper.Variables
    depending on the l'th measurement oₗ are stored in arrays of lengt L (since l = 1,...,L), where each 
    entry corresponds to the value corresponding to a measurement oₗ. Furthermore, we store 
    prod_{k'=1}^{k} {oₗ[k'] ⊳ pₘ[k'] in the variable A, w(oₗ) in "weight" and w¬ₖ(oₗ) as w. To try out 
    different Paulis (or create 3^{-w}), we copy to A_ and w_.

    An example for a set of observables would be :
        observables = String["Y" "Y" "Y" "X"; "Z" "Z" "Z" "X"; "I" "Z" "Y" "Y"]"""

    ϵ = precision 
    ν = 1 - exp(-ϵ^2/2)
    M = n_measurements
    L = size(observables)[1]
    n_qubits = size(observables)[2]
    Paulis = String["X" "Y" "Z"]
    measurements = Array{String, 2}(undef, M, n_qubits)

    function cost_func(h_vec, A_vec, w_vec, C_vec, m_idx)
        w_vec_ = copy(w_vec)
        for l in 1:L 
            w_vec_[l] = 3^(-w_vec_[l])
        end
        cost = exp.(-ϵ^2/2 .* h_vec) .* (1 .- ν .* A_vec .* w_vec_) .* (C_vec.^(M-m_idx))
        return sum(cost)
    end

    # compute C = 1 - ν3^(-w(oₗ))
    weight = zeros(Float64, L)
    for l in 1:L
        for k in 1:n_qubits
            if observables[l,k] != "I"
                weight[l] += 1
            end
        end
    end
    weight_ = copy(weight)
    for l in 1:L 
        weight_[l] = 3^(-weight[l])
    end
    C = 1 .- ν .* weight_
    
    h = zeros(Float64, L) # hit count per observable
    # loop over measurements
    for m in 1:M
        A = ones(Float64, L)
        w = copy(weight) 
        for k in 1:n_qubits
            # update partial weights
            for l in 1:L 
                if observables[l,k] != "I"
                    w[l] -= 1
                end
            end
            # find best next Pauli
            cost_vec = zeros(Float64, 3)
            for i in 1:3
                A_ = copy(A)
                for l in 1:L
                    if !(observables[l,k] == "I" || observables[l,k] == Paulis[i])
                        A_[l] *= 0
                    end
                end 
                cost_vec[i] = cost_func(h, A_, w, C, m)
            end
            # update everything with the correct Pauli string 
            measurements[m,k] = Paulis[argmin(cost_vec)]
            for l in 1:L 
                if !(observables[l,k] == "I" || observables[l,k] == Paulis[argmin(cost_vec)])
                    A[l] *= 0
                end
            end
        end
        h += A 
    end
    return measurements#, sum(exp.(-ϵ^2/2 .* h))
end

# Do the counting stuff here, this will also serve as a comparison
function prediction_Pauli_basis(measurement_outcomes, measurement_observables, observables)
    M = size(measurements_outcomes)[1]
    N = size(measurements_outcomes)[2]
    hit_count = 0 
    sum_prod = 0
    
    function is_hitting(observable_1, observable_2)
        hit = true 
        for n in 1:N
            if !(observable_1[n] == observable_2[n] || observable_1[n] == "I")
                hit = false
            end 
        end
        return hit
    end

    for m in 1:M
        if is_hitting(observables[m,:], measurement_observables[m,:])
            hit_count += 1
            sum_prod += prod(measurement_outcomes[m,:])
        end 
    end

    return sum_prod / hit_count
end