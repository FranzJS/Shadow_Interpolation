# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Julia 1.7.2
#     language: julia
#     name: julia-1.7
# ---

using Pkg
Pkg.activate("shadow")
include("../numerics/random_clifford.jl")
using LinearAlgebra

# We will explore the usage of the different functions in "random_clifford.jl". For theoretical background and further explanation of the algorithms, see https://arxiv.org/abs/2008.06011.
#
# The main functionality of "random_cliford.jl" is to provide a sampling scheme for random Clifford gates. To sample a random Clifford, we just have to pick the system size in qubits.

n_qubits = 2
random_clifford = sample_random_Clifford(n_qubits)

# What we get is a list of tuples, each tuple detailing a gate from the generating set of the Cliffords (Hadamard $H$, Phase $S$, CX (CNOT) $CX$). The numbers datail which qubits the gates act on (for the $CX$-gate, the first qubit is the controll qubit). When executed in succession, these gates implement the random Clifford gate. We can also transform this list into an explicit matrix representation.

matrix_representation = Clifford_circuit_to_matrix(random_clifford, n_qubits)

# Note that this transformation is very slow and should ideally only be used for prototyping.
#
# To test that our algorithm works the way we intend to, we can check if all Gates are generated with equal probability. An easy (although not exhaustive) test is to count the Hadamard gates that occur when sampling the single qubit group (should be $\frac{1}{24}$).

# +
n_qubits = 1
H = 1/sqrt(2)*[1 1;1 -1]
samples = 1_000_000
cnt = 0

Threads.@threads for i in 1:samples # embarrassingly parallel
    global H
    global cnt
    local x
    x = Clifford_circuit_to_matrix(sample_random_Clifford(n_qubits), n_qubits)
    if norm(x-H) < 1e-10 
        cnt += 1
    end
end

println("expected ration:", 1/number_of_Cliffords(n_qubits))
println("observed ratio:", cnt/samples)
# -

# Now we will go in more depth through the details of the implementation.
# First, we will get accustomed with the bit representation of a Pauli string. Since $XZ = -iY$, we can represent a Pauli string via two bitstrings $x,z$
# \begin{align}
#     P(x,y) = \prod_{j=1}^n i^{x_j z_j} X_j Z_j
# \end{align}
# where $X_j$ applies $X$ on the $j$'th qubit and identities on all other qubits ($Z_j$ is defined likewise). 
#
# We thus represent a Pauli string as a BitArray [x, z]. Some examples:

XZX = BitArray([1 0 1; 0 1 0])
XYX = BitArray([1 1 1; 0 1 0])
XYZ = BitArray([1 1 0; 0 1 1]);

# Note that we do not care about phase at this point, this is due to a technical reason which we'll point out later. In principle, one can also use an additional bit $s$ to include a phase as $(-1)^s$.

# +
# ToDo: Continue...
# -


