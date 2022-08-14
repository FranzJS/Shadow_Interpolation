function vec_mod(x)
    n = length(x)
    for i in 1:n
        x[i] = mod(x[i], 0:1)
    end
    return x
end


function Zn(n)
    Z = [1 0; 0 -1]
    r = 1
    for i in 1:n
        r = kron(r,Z)
    end
    return r
end


function fourier_basis(t, k; period=2*pi, shift=0)
    return exp(1im*2*pi*(k.-shift)*t/period)
end


function FourierSeries(coefficients, period)
    N = size(coefficients)[1]
    m = Int((N-1)/2) 
    function fourierSeries(t)
        sum = zeros(Complex, size(t)[1])
        for k in -m:m
            sum .+= coefficients[k+m+1]*exp.(1im*2*pi/period*k.*t)
        end
        return real(sum)
    end
    return fourierSeries  
end


function positive(a)
    if a < 0
        return 0
    else
        return a
    end
end

function negative(a)
    if a > 0
        return 0
    else
        return abs(a)
    end 
end

""" Split A+ into positive real, negative real, positive imaginary and negative imaginary part. """
function matrix_split(A)
    A_real_positive = positive.(real.(A))
    A_real_negative = negative.(real.(A))
    A_imag_positive = positive.(imag.(A))
    A_imag_negative = negative.(imag.(A))
    return (A_real_positive, A_real_negative, A_imag_positive, A_imag_negative)
end


""" Normalized sinc function. """
function sinc(x)
    if abs(x) < 1e-12
        return 1
    else
        return sin(x)/x
    end
end


function pauli_string_to_tensor(pauli_string)
    U = 1
    for P in pauli_string
        if P == 'X'
            U = kron(U, [0 1; 1 0])
        elseif P == 'Y'
            U = kron(U, [0 -1im; 1im 0])
        elseif P == 'Z'
            U = kron(U, [1 0; 0 -1])
        elseif P == 'I'
            U = kron(U, [1 0; 0 1])
        end
    end
    return U
end



function sparse_indices(x; ϵ=1e-10)
    n = size(x)[1]
    sparse_indices = Int64[]
    for i in 1:n
        if abs(x[i]) > 1e-10
            push!(sparse_indices, i)
        end
    end
    return sparse_indices
end


""" Return a matrix B which arises the columns of A specified in indices. """
function reduce_A(A, indices)
    m, n = size(A)
    k = size(indices)[1]
    B = Matrix{Complex{Float64}}(undef, m, k)
    for (i, idx) in enumerate(indices)
        B[1:end, i] = A[1:end, idx]
    end
    return B
end


""" Construct a vector of dimension dim, where the non-zero entries are the indices from x at entries 
specified in indices. """
function zero_pad(x, indices, dim)
    z = zeros(Complex{Float64}, dim)
    for (i, idx) in enumerate(indices)
        z[idx] = x[i]
    end
    return z
end


function hamiltonian_frequencies_to_data_frequencies(ham_freqs)
    data_freqs = Int64[]
    dim = size(ham_freqs)[1]
    for i in 1:dim
        for j in 1:dim
            push!(data_freqs, ham_freqs[i]-ham_freqs[j])
        end 
    end
    return sort(unique(data_freqs))
end