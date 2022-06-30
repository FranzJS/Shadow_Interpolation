import StatsBase 
using Test

# test measurement_procedure.jl 
include("../measurement_procedure.jl")

# test random_measurement_procedure()
measurement = random_measurement_procedure(5000, 5000)
counting_statistics = StatsBase.countmap(measurement)

for i in ["X", "Y", "Z"]
    @test isapprox(counting_statistics[i]/length(measurement), 1/3, atol=1e-03)
end

print("Test measurement_procedure.random_measurement_procedure() passed!")