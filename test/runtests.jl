using Test
include "testProblemUtils.jl"

@testset "checkProblem tests" begin
    test_checkProblem()
    #test_checkSample()
    test_checkData()
    test_checkGamma()
    test_loss()
    test_risk()
end
