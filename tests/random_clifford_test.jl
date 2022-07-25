using Test
using Printf
using ProgressBars

include("../numerics/random_clifford.jl")
println("start testing")
############################################
##### test: check_anticommuting() ##########
############################################
println("test: check_anticommuting()")
# some examples 
@test check_anticommuting(BitArray([1 0 1; 0 1 0]), BitArray([0 1 0; 1 0 1])) 

@test check_anticommuting(BitArray([0 1 0 1; 0 1 1 0]), BitArray([1 0 1 0; 1 1 1 1]))

"""Simple (readable) algorithm to check for anticommuting Pauli strings. About 5 times slower than check_anticommuting()."""
function simple_check_anticommuting(Pauli_1, Pauli_2)
    n = size(Pauli_1)[2]
    anticommuting_Pauli_cnt = 0
    for i in 1:n
        if (Pauli_1[1:2, i] == [0,0]) | (Pauli_2[1:2, i] == [0,0])
            continue
        elseif Pauli_1[1:2, i] != Pauli_2[1:2, i]
            anticommuting_Pauli_cnt += 1
        end 
    end 
    if anticommuting_Pauli_cnt % 2 == 1
        return true
    else 
        return false
    end
end

function test_check_anticommuting(samples)
    x = rand(1:1000,samples)
    for i in 1:1000
        P = bitrand(2,x[i])
        Q = bitrand(2,x[i])
        @test check_anticommuting(P,Q) == simple_check_anticommuting(P,Q)
    end
end
    
test_check_anticommuting(100000)

# One can also check if the right statistics are reproduced by check_anticommuting()


#######################################
#### test sample_random_Clifford() ####
#######################################

""" 
On a single device, all tests with n_qubits > 1 require hours in runtime.

For n_qubits=2 and samples=300_000_000 one: cnt/samples=8.659666666666666e-5
and 1/number_of_Cliffords(2)=8.680555555555556e-5.
"""
function test_sample_statistics(samples, n_qubits, atol=1e-03)
    reference_Clifford = Clifford_circuit_to_matrix(sample_random_Clifford(n_qubits), n_qubits)
    cnt = 0
    Threads.@threads for i in ProgressBar(1:samples)
        x = sample_random_Clifford(n_qubits)
        if norm(Clifford_circuit_to_matrix(x, n_qubits) - reference_Clifford) < 1e-10
            cnt += 1
        end
    end
    println("cnt/samples:", cnt/samples)
    println("real ratio:", 1/number_of_Cliffords(n_qubits))
    return norm(cnt/samples - 1/number_of_Cliffords(n_qubits)) < atol
end


n_run = 1 # for a thorough test, set this value high (and try also for n_qubits=2)
for i_run in ProgressBar(1:n_run)
    @test test_sample_statistics(1_000_000, 1, 1e-03)  
end



function test_single_qubit_Clifford(samples)
    # note the definition of the Clifford group in e.g. http://home.lu.lv/~sd20008/papers/essays/Clifford%20group%20[paper].pdf
    # basically mapping from P as given below to P is enough to define the Clifford group
    P = Dict("+X" => [0 1; 1 0], "+Y" => [0 -1im; 1im 0], "+Z" => [1 0; 0 -1], "-X" => -1*[0 1; 1 0], "-Y" => -1*[0 -1im; 1im 0], "-Z" => -1*[1 0; 0 -1])
    cnt = 0 # counts conjugations to the set P
    for i in ProgressBar(1:samples)
        x = sample_random_Clifford(1)
        x = Clifford_circuit_to_matrix(x, 1)
        for key in keys(P)
            y = x*P[key]*x' # conjugate for all elements in P
            #println(y)
            for key in keys(P)
                if norm(P[key]-y) < 1e-10 # test if still in P
                    cnt += 1
                    #println("correct!")
                end
            end
        end
    end
    return cnt == 6*samples # for every sample, there are 6 conjugations which all should still be in P
end


@test test_single_qubit_Clifford(100_000)




