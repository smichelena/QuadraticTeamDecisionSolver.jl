using Test

include("../src/quadTeamProblems.jl")


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

function test_checkSample()
    p = QuadTeamProblem(2, [2, 3], [4, 5], Float64)

    # Test case where sample types and dimensions are correct
    s1 = Sample([rand(2), rand(3)], rand(9, 9), rand(9), rand())
    @test checkSample(p, s1) == s1

    # Test case where measurement dimensions don't match
    s2 = Sample([rand(2), rand(5)], rand(9, 9), rand(9), rand())
    @test_throws AssertionError checkSample(p, s2)

    # Test case where R dimensions don't match
    s3 = Sample([rand(2), rand(3)], rand(2, 2), rand(9), rand())
    @test_throws AssertionError checkSample(p, s3)

    # Test case where r dimensions don't match
    s4 = Sample([rand(2), rand(3)], rand(9, 9), rand(12), rand())
    @test_throws AssertionError checkSample(p, s4)

    # Test case where sample type is wrong
    s5 = Sample(
        [rand(Int32, 2), rand(Int32, 3)],
        rand(Int32, 9, 9),
        rand(Int32, 9),
        rand(Int32),
    )
    @test_throws AssertionError checkSample(p, s5)

end

function test_checkData()
    p = QuadTeamProblem(2, [2, 3], [4, 5], Float64)

    # Test case where all samples pass the check
    S1 = [
        Sample([rand(2), rand(3)], rand(9, 9), rand(9), rand()),
        Sample([rand(2), rand(3)], rand(9, 9), rand(9), rand()),
    ]
    @test checkData(p, S1) == S1

    # Test case where one sample fails the check
    S2 = [
        Sample([rand(2), rand(3)], rand(9, 9), rand(9), rand()),
        Sample([rand(2), rand(3)], rand(4, 9), rand(9), rand()), #wrong R
    ]
    @test_throws AssertionError checkData(p, S2)
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

function test_loss()
    P = QuadTeamProblem(2, [2, 3], [4, 5], Float64)

    # Test case where loss should return a valid result
    Y = [rand(2), rand(3)]
    R = rand(9, 9)
    r = rand(9)
    c = 1.0

    γ₁(x) = [x[1], x[1], x[1], x[1]]
    γ₂(x) = [x[1], x[2], x[3], x[1], x[2]]

    γ = [γ₁, γ₂]

    sample = Sample(Y, R, r, c)

    @test loss(sample, γ) isa Real
end

function test_risk()
    # Define a sample set
    S = [
        Sample([rand(2), rand(3)], rand(4, 4), rand(4), rand()),
        Sample([rand(2), rand(3)], rand(4, 4), rand(4), rand()),
    ]

    # Define a vector of functions
    γ = [
        x -> sum(x), #output dim = 1
        x -> x .* 2.0, #output dim = dim(Y2) = 3
    ]

    # Calculate the risk
    r = risk(S, γ)

    # Expected risk value
    expected_risk = (loss(S[1], γ) + loss(S[2], γ)) / length(S)

    # Compare the calculated risk with the expected value
    @test isapprox(r, expected_risk)
end
