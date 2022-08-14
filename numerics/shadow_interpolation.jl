using StatsBase
using LinearAlgebra
include("../numerics/utils.jl")

# ToDo: median of means
"""Note: measurement_primitive == "pauli" is still untested!"""
function TraceEstimator(measurement_func, shadows, probabilities, samples; measurement_primitive="clifford")
    n_qubits = Int((size(shadows)[3] -1)/2)
    N, n, _, _ = size(shadows)
    shadow_indices = Vector{Int32}(undef, samples)
    snapshot_indices = Vector{Int32}(undef, samples)
    sample!(1:N, pweights(probabilities), shadow_indices)
    sample!(1:n, pweights(ones(n)), snapshot_indices)
    outcome = 0
    for i_sample in 1:samples
        outcome += measurement_func(shadows[shadow_indices[i_sample], snapshot_indices[i_sample], 1:end, 1:end])
    end
    outcome /= samples
    if measurement_primitive == "clifford"
        outcome *= (2^n_qubits +1)
    elseif measurement_primitive == "pauli"
        outcome *= 3^n_qubits
    end
    return outcome
end



function ShadowShannonInterpolation(measurement_func, shadows, period; n_samples=100_000, baseline_shift=true)
    N = Int((size(shadows)[1] - 1)/2)
    h = period/N/2
    trace_coeffs = Vector{Float64}(undef, 2*N+1)
    Threads.@threads for n in -N:N
        weights = zeros(2*N+1)
        weights[n+N+1] = 1
        trace_coeffs[n+N+1] = TraceEstimator(measurement_func, shadows, weights, n_samples)
    end
    shift = 0
    if baseline_shift == true
        shift = (trace_coeffs[1] + trace_coeffs[end]) / 2
        trace_coeffs = trace_coeffs .- shift
    end
    
    function Interpoliant(t)
        sum = zero(t)
        for n in -N:N
            sum += trace_coeffs[n+N+1] * sinc.(pi/h*(t .- n*h))
        end
        return sum .+ shift
    end
    return Interpoliant
end



""" basis_function must have arguments "data_points" (must accept vectors here) and
the number of the basis function k (as in e^{ikt})."""
function system_matrix(time_points, n_basis_functions, basis_function)
    N = size(time_points)[1]
    A = Matrix{Complex{Float64}}(undef, N, n_basis_functions)
    for k in 0:n_basis_functions-1
        A[1:end, k+1] = basis_function.(time_points, k)
    end
    return A
end

""" For the solution of Ax=b, supply e.g. the pseudo inverse of A as inv_system_matrix. """
function ShadowBestapproximation(measurement_func, shadows, inv_system_matrix; n_samples=100_000)
    n_basis_functions = size(inv_system_matrix)[1]
    w_real_positive, w_real_negative, w_imag_positive, w_imag_negative = matrix_split(inv_system_matrix)
    coefficients = Vector{Complex{Float64}}(undef, n_basis_functions)
    for k in 1:n_basis_functions
        z_real_pos = sum(w_real_positive[k, 1:end])
        z_real_neg = sum(w_real_negative[k, 1:end])
        z_imag_pos = sum(w_imag_positive[k, 1:end])
        z_imag_neg = sum(w_imag_negative[k, 1:end])
        
        if z_real_pos == 0
            c_real_pos = 0
        else
            c_real_pos = TraceEstimator(measurement_func, shadows, w_real_positive[k, 1:end], n_samples)
        end
        
        if z_real_neg == 0
            c_real_neg = 0
        else
            c_real_neg = TraceEstimator(measurement_func, shadows, w_real_negative[k, 1:end], n_samples)
        end
        
        if z_imag_pos == 0
            c_imag_pos = 0
        else
            c_imag_pos = TraceEstimator(measurement_func, shadows, w_imag_positive[k, 1:end], n_samples)
        end
        
        if z_imag_neg == 0
            c_imag_neg = 0
        else
            c_imag_neg = TraceEstimator(measurement_func, shadows, w_imag_negative[k, 1:end], n_samples)
        end
        

        coefficients[k] = z_real_pos*c_real_pos-z_real_neg*c_real_neg + 1im*(z_imag_pos*c_imag_pos-z_imag_neg*c_imag_neg)
    end
    return coefficients
end
