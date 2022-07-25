using StatsBase
using LinearAlgebra
include("../numerics/utils.jl")


# ToDo: median of means
# Careful, only supports n_qubit_Clifford for now (2^n_qubits +1)
function TraceEstimator(measurement_func, shadows, probabilities, samples)
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
    outcome *= (2^n_qubits +1)
    return outcome
end



function ShadowShannonInterpolation_(measurement_func, shadows, period; n_samples=100_000, baseline_shift=true)
    N = Int((size(shadows)[1] - 1)/2)
    h = period/N/2
    trace_coeffs = Vector{Float64}(undef, 2*N+1)
    Threads.@threads for n in -N:N
        # this is not perfectly efficient
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