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