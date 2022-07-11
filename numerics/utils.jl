function vec_mod(x)
    n = length(x)
    for i in 1:n
        x[i] = mod(x[i], 0:1)
    end
    return x
end