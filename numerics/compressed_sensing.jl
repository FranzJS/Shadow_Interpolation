include("../numerics/utils.jl")


""" Iterative Hard Thresholding. Right now, the system is normalized with 1/sqrt(n), maybe 
this is not universally a good idea?"""
function IHT(A, b, s; max_iter=1000, breaking=false)
    iter = 0
    n = size(A)[2]
    A /= sqrt(n)
    b /= sqrt(n)
    x = zeros(n)
    p = n-s
    mask_old = []
    println(opnorm(A))
    while iter <= max_iter
        iter += 1
        x = x + A'*(b-A*x)
        mask_new = partialsortperm(abs.(x), 1:p)
        x[mask_new] .= 0
        if breaking == true
            if mask_new == mask_old
                println("break")
                break
            end
        end
        mask_old = copy(mask_new)
    end
    return x
end



""" Return the indices of the s largest entries. """
function HTP_S(x, s)
    b = partialsortperm(abs.(x), 1:s, rev=true)
    return sort(b)
end

    
    
"""Hard Thresholding Pursuit."""
function HTP(A, b, sparsity; max_iter=1_000, breaking=false)
    iter = 0
    n = size(A)[2]
    A /= sqrt(n)
    b /= sqrt(n)
    x = zeros(n)
    S_old = []
    while iter <= max_iter
        iter += 1
        z = x + A'*(y-A*x)
        S_new = HTP_S(z, sparsity)
        A_sparse = reduce_A(A, S_new)
        x_ = A_sparse\b
        x = zero_pad(x_, S_new, n)
        
        # stopping criterion
        if breaking == true
            if S_new == S_old
                println("break")
                break
            end
        end
        S_old = copy(S_new)
    end
    return x
end