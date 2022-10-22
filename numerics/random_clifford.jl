using Random
using LinearAlgebra
# For better understanding of the algorithms check https://arxiv.org/abs/2008.06011


###############################
#### Some helper functions ####
###############################

"""Compute the number of elements in the n-qubit Clifford group."""
function number_of_Cliffords(n)
    prod = 1
    for j in 1:n
        prod *= 4^j - 1
    end
    return 2^(n^2 + 2n) * prod
end

""" This is terribly inefficient. Just for prototyping. """
function Clifford_gate_to_matrix(Clifford_gate, n_qubits)
    M = Dict("I" => [1 0; 0 1], "X" => [0 1; 1 0], "Y" => [0 -1im; 1im 0], "Z" => [1 0; 0 -1], "H" => 1/sqrt(2)*[1 1; 1 -1], "S" => [1 0; 0 1im])
    M["P0"], M["P1"] = [1 0; 0 0], [0 0; 0 1]

    matrix_representation = 1
    if Clifford_gate[1] == "CX"
        if Clifford_gate[2] < Clifford_gate[3]
            for i_qubit in 1:Clifford_gate[2]-1
                matrix_representation = LinearAlgebra.kron(matrix_representation, M["I"])
            end
            matrix_representation_1 = LinearAlgebra.kron(matrix_representation, M["P0"])
            matrix_representation_2 = LinearAlgebra.kron(matrix_representation, M["P1"])

            for i_qubit in Clifford_gate[2]+1:Clifford_gate[3]-1
                matrix_representation_1 = LinearAlgebra.kron(matrix_representation_1, M["I"])
                matrix_representation_2 = LinearAlgebra.kron(matrix_representation_2, M["I"])
            end
            matrix_representation_1 = LinearAlgebra.kron(matrix_representation_1, M["I"])
            matrix_representation_2 = LinearAlgebra.kron(matrix_representation_2, M["X"])

            for i_qubit in Clifford_gate[3]+1:n_qubits
                matrix_representation_1 = LinearAlgebra.kron(matrix_representation_1, M["I"])
                matrix_representation_2 = LinearAlgebra.kron(matrix_representation_2, M["I"])
            end
            matrix_representation = matrix_representation_1 + matrix_representation_2
        else
            for i_qubit in 1:Clifford_gate[3]-1
                matrix_representation = LinearAlgebra.kron(matrix_representation, M["I"])
            end
            matrix_representation_1 = LinearAlgebra.kron(matrix_representation, M["I"])
            matrix_representation_2 = LinearAlgebra.kron(matrix_representation, M["X"])

            for i_qubit in Clifford_gate[3]+1:Clifford_gate[2]-1
                matrix_representation_1 = LinearAlgebra.kron(matrix_representation_1, M["I"])
                matrix_representation_2 = LinearAlgebra.kron(matrix_representation_2, M["I"])
            end
            matrix_representation_1 = LinearAlgebra.kron(matrix_representation_1, M["P0"])
            matrix_representation_2 = LinearAlgebra.kron(matrix_representation_2, M["P1"])
            for i_qubit in Clifford_gate[2]+1:n_qubits
                matrix_representation_1 = LinearAlgebra.kron(matrix_representation_1, M["I"])
                matrix_representation_2 = LinearAlgebra.kron(matrix_representation_2, M["I"])
            end
        matrix_representation = matrix_representation_1 + matrix_representation_2
        end
        return matrix_representation
    else
        for i_qubit in 1:Clifford_gate[2]-1
            matrix_representation = LinearAlgebra.kron(matrix_representation, M["I"])
        end
        matrix_representation = LinearAlgebra.kron(matrix_representation, M[Clifford_gate[1]])
        for i_qubit in Clifford_gate[2]+1:n_qubits
            matrix_representation = LinearAlgebra.kron(matrix_representation, M["I"])
        end
        return matrix_representation
    end
end

""" Terribly slow! Just for prototyping. """
function Clifford_circuit_to_matrix(Clifford_gates, n_qubits)
    n = size(Clifford_gates)[1]
    circuit = 1
    for i in 1:n
        circuit = circuit * Clifford_gate_to_matrix(Clifford_gates[n+1-i], n_qubits)
    end
    return circuit
end





################################################
#### Buidling blocks of the final algorithm ####
################################################

"""
    check_anticommuting(Pauli_1, Pauli_2)

Check if two Pauli strings given in bit notation [X bits; Z bits] anticommute. Note that only unsigned Pauli
strings can be passed.
# Examples (XZX and ZXZ)
```julia-repl
julia> P_1 = BitArray([1 0 1; 0 1 0])
julia> P_2 = BitArray([0 1 0; 1 0 1])
julia> check_anticommuting(P_1, P_2)
true
```
"""
function check_anticommuting(Pauli_1, Pauli_2)
    # 1) filter out identities (Paulis commute with identity)
    # 1.1) create array with 1 on every qubit with an identity in at least on Pauli, else 0
    identity_filter = BitArray((Pauli_1[1,1:end] + Pauli_1[2, 1:end] .== 0) .| (Pauli_2[1,1:end] + Pauli_2[2, 1:end] .== 0))

    # 1.2) update Pauli bitstrings to only contain positions relevant for anticommuting (no identities)
    P_11 = Pauli_1[1, 1:end][identity_filter .== 0]
    P_12 = Pauli_1[2, 1:end][identity_filter .== 0]
    P_21 = Pauli_2[1, 1:end][identity_filter .== 0]
    P_22 = Pauli_2[2, 1:end][identity_filter .== 0]
    # note that while Pauli is a 2xN matrix, P is a Nx1 vector. The transpose is needed to regain the old structure of Pauli
    Pauli_1 = vcat(transpose(P_11), transpose(P_12))
    Pauli_2 = vcat(transpose(P_21), transpose(P_22))

    # filter out same Paulis (Paulis commute with themselves)
    x_comp = BitArray(Pauli_1[1, 1:end] .== Pauli_2[1, 1:end]) # if X bit on qubit same 1, else 0
    z_comp = BitArray(Pauli_1[2, 1:end] .== Pauli_2[2, 1:end]) # if Z bit on qubit same 1, else 0
    x_and_z_comp = BitArray(x_comp + z_comp .!= 2) # if X and Z bit NOT same, then 1, else 0 (same Pauli iff same X and Z bit)
    non_commuting_Paulis_cnt = sum(x_and_z_comp)
    # the whole string anticommutes for an odd number of anticommuting Paulis
    if non_commuting_Paulis_cnt % 2 == 1
        return true
    else
        return false
    end
end


"""
    apply_H!(i_qubit, Pauli_subtableau)

Apply a H gate to a Pauli subtableau. Pauli subtableau has to be given in the notation
[X bits Pauli 1, Z bits Pauli 1, X bits Pauli 2, Z bits Pauli 2]. No sign information is
passed here.
"""
function apply_H!(i_qubit, Pauli_subtableau, l)
    x1, x2 = Pauli_subtableau[1, i_qubit], Pauli_subtableau[3, i_qubit]
    Pauli_subtableau[1, i_qubit], Pauli_subtableau[3, i_qubit] = Pauli_subtableau[2, i_qubit], Pauli_subtableau[4, i_qubit]
    Pauli_subtableau[2, i_qubit], Pauli_subtableau[4, i_qubit] = x1, x2
    return ("H", i_qubit+(l-1))
end


"""
    apply_S!(i_qubit, Pauli_subtableau)

Apply a S gate to a Pauli subtableau. Pauli subtableau has to be given in the notation
[X bits Pauli 1, Z bits Pauli 1, X bits Pauli 2, Z bits Pauli 2]. No sign information is
passed here.
"""
function apply_S!(i_qubit, Pauli_subtableau, l)
    Pauli_subtableau[2, i_qubit] = mod(Pauli_subtableau[1, i_qubit] + Pauli_subtableau[2, i_qubit], 0:1)
    Pauli_subtableau[4, i_qubit] = mod(Pauli_subtableau[3, i_qubit] + Pauli_subtableau[4, i_qubit], 0:1)
    return ("S", i_qubit+(l-1))
end


"""
    apply_CX!(i_qubit, j_qubit, Pauli_subtableau)

Apply a CX (CNOT) gate to a Pauli subtableau. Pauli subtableau has to be given in the notation
[X bits Pauli 1, Z bits Pauli 1, X bits Pauli 2, Z bits Pauli 2]. No sign information is
passed here.
"""
function apply_CX!(i_qubit, j_qubit, Pauli_subtableau, l)
    Pauli_subtableau[1, j_qubit] = mod(Pauli_subtableau[1, i_qubit] + Pauli_subtableau[1, j_qubit], 0:1) 
    Pauli_subtableau[3, j_qubit] = mod(Pauli_subtableau[3, i_qubit] + Pauli_subtableau[3, j_qubit], 0:1) 

    Pauli_subtableau[2, i_qubit] = mod(Pauli_subtableau[2, j_qubit] + Pauli_subtableau[2, i_qubit], 0:1) 
    Pauli_subtableau[4, i_qubit] = mod(Pauli_subtableau[4, j_qubit] + Pauli_subtableau[4, i_qubit], 0:1) 
    return ("CX", i_qubit+(l-1), j_qubit+(l-1))
end



function sweep_step_1!(Pauli_subtableau, Clifford_gates, l, Pauli_idx_start=1)
    n_qubit = size(Pauli_subtableau)[2]
    p = Pauli_idx_start
    for i_qubit in 1:n_qubit
        if Pauli_subtableau[p+1, i_qubit] == 1
            if Pauli_subtableau[p, i_qubit] == 0
                push!(Clifford_gates, apply_H!(i_qubit, Pauli_subtableau, l))
            else
                push!(Clifford_gates, apply_S!(i_qubit, Pauli_subtableau, l))
            end
        end
    end
end


function create_sorted_list(Pauli_subtableau, Pauli_idx_start)
    n_qubit = size(Pauli_subtableau)[2]
    p = Pauli_idx_start
    sorted_list = Int[]
    for i_qubit in 1:n_qubit
        if Pauli_subtableau[p, i_qubit] == 1
            push!(sorted_list, i_qubit)
        end
    end
    return sorted_list
end


function update_sorted_list(sorted_list)
    n = length(sorted_list)
    mask = BitArray([])
    for i in 1:n
        push!(mask, mod(i, 0:1))
    end
    return sorted_list[mask]
end

function sweep_step_23!(Pauli_subtableau, Clifford_gates, l, Pauli_idx_start=1)
    sorted_list = create_sorted_list(Pauli_subtableau, Pauli_idx_start)
    while length(sorted_list) > 1
        for i in 1:(length(sorted_list)-1)
            push!(Clifford_gates, apply_CX!(sorted_list[i], sorted_list[i+1], Pauli_subtableau, l))
        end
        sorted_list = update_sorted_list(sorted_list)
    end
    if sorted_list[1] != 1
        push!(Clifford_gates, apply_CX!(1, sorted_list[1], Pauli_subtableau, l))
        push!(Clifford_gates, apply_CX!(sorted_list[1], 1, Pauli_subtableau, l))
        push!(Clifford_gates, apply_CX!(1, sorted_list[1], Pauli_subtableau, l))
    end
end


function sweep_step_4!(Pauli_subtableau, Clifford_gates, l)
    push!(Clifford_gates, apply_H!(1, Pauli_subtableau, l))
    sweep_step_1!(Pauli_subtableau, Clifford_gates, l, 3)
    sweep_step_23!(Pauli_subtableau, Clifford_gates, l, 3)
    push!(Clifford_gates, apply_H!(1, Pauli_subtableau, l))
end


function sweep_step_5!(Clifford_gates, l)
    s = bitrand(2)
    if (s[1] == 0) & (s[2] == 1)
        push!(Clifford_gates, ("X", l))
    elseif (s[1] == 1) & (s[2] == 0)
        push!(Clifford_gates, ("Z", l))
    elseif (s[1] == 1) & (s[2] == 1)
        push!(Clifford_gates, ("Y", l))
    end
end


#########################################
#### Sample random Clifford matrices ####
#########################################

function sample_random_Clifford(n_qubits)
    Clifford_gates = [] # list of Clifford gates applied so far
    for l in 1:n_qubits
        # 1) sample two random anticommuting Paulis
        Pauli_subtableau = BitArray(undef, (4,n_qubits-(l-1)))
        while true
            P,Q = bitrand(2,n_qubits-(l-1)), bitrand(2,n_qubits-(l-1))
            if check_anticommuting(P,Q) == true
                Pauli_subtableau[1:2, 1:end] = P
                Pauli_subtableau[3:4, 1:end] = Q
                break
            end
        end
        # 2) sweep rows 2i, 2i-1
        sweep_step_1!(Pauli_subtableau, Clifford_gates, l)
        sweep_step_23!(Pauli_subtableau, Clifford_gates, l)
        sweep_step_4!(Pauli_subtableau, Clifford_gates, l)
        sweep_step_5!(Clifford_gates, l)
    end
    return Clifford_gates
end



