using LinearAlgebra

function time_evolution(state, diag_hamiltonian, t)
    Ut = exp(-1im*diag_hamiltonian*t)
    return Ut * state * Ut'
end


""" Pass a vector t."""
function create_time_series(state, diag_hamiltonian, t)
    m = size(t)[1]
    n = size(state)[1]
    x = Array{Complex{Float64}}(undef, m, n, n)
    for i in 1:m
        x[i, 1:end, 1:end] = time_evolution(state, diag_hamiltonian, t[i])
    end
    return x
end