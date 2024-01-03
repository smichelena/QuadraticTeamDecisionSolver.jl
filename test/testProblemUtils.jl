using Test
using QuadraticTeamDecisionSolver

function test_checkProblem()
    # Test case where problem specification is correct
    p = QuadTeamProblem(2, [2, 3], [4, 5], Float64)
    @test checkProblem(p) == p

    # Test case where measurement dimension sizes don't match
    p = QuadTeamProblem(2, [2, 3, 4], [4, 5], Float64)
    @test_throws AssertionError checkProblem(p)

    # Test case where action space dimension sizes don't match
    p = QuadTeamProblem(2, [2, 3], [4, 5, 6], Float64)
    @test_throws AssertionError checkProblem(p)

    # Test case where number of agents doesn't match
    p = QuadTeamProblem(3, [2, 3], [4, 5], Float64)
    @test_throws AssertionError checkProblem(p)
end

function test_checkGamma()
    P = QuadTeamProblem(2, [2, 3], [4, 5], Float64)

    # Test case where checkGamma should return true
    γ₁¹(x) = [x[1], x[1], x[1], x[1]]
    γ₂¹(x) = [x[1], x[1], x[1], x[1], x[1]]

    γ¹ = [γ₁¹, γ₂¹]

    @test checkGamma(P, γ¹) == γ¹

    # Test case where checkGamma should return false
    γ₁²(x) = [2 * x[1]]
    γ₂²(x) = [x[1]]

    γ² = [γ₁², γ₂²]

    @test_throws AssertionError checkGamma(P, γ²)
end