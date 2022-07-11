using Test 
using ProgressBars
include("../numerics/classical_shadow.jl")


println("new")

# test: measure_computational_basis(); measure_computational_basis_()

###################################
###### EPR-State measurement ######
###################################

# later on a systematic test should be added


""" Measuring the EPR-state in the computational basis should result in 
either 00 or 11 with equal probability. """

samples = 1_000_000

# test: measure_computational_basis()
function test_measure_computational_basis(samples; precision=1e-03)
    EPR_state = [1/sqrt(2), 0, 0, 1/sqrt(2)]
    EPR_density_matrix = EPR_state * EPR_state'

    cnt_00 = 0
    cnt_11 = 0
    for i in 1:samples 
        x = measure_computational_basis(diag(EPR_density_matrix))
        if x == "00"
            cnt_00 += 1
        elseif x == "11"
            cnt_11 += 1
        end
    end
    return (abs(0.5-cnt_00/samples) < precision) && (abs(0.5-cnt_11/samples) < precision)
end


# test: measure_computational_basis_()
function test_measure_computational_basis_(samples; precision=1e-03)
    #tableau form: 1:n row: destabilizers, n+1:2n row: stabilizers, last column: phases, 
    #last row: convenient storage for technical reasons
    cnt_00 = 0
    cnt_11 = 0
    for i in 1:samples
        EPR_tableau = BitArray([0 1 0 0 0; 0 0 1 0 0; 1 1 0 0 0; 0 0 1 1 0; 0 0 0 0 0])
        x = measure_computational_basis_(EPR_tableau)
        if x == "00"
            cnt_00 += 1
        elseif x == "11"
            cnt_11 += 1
        end
    end
    return (abs(0.5-cnt_00/samples) < precision) && (abs(0.5-cnt_11/samples) < precision)
end

@test test_measure_computational_basis(samples)
@test test_measure_computational_basis_(samples)