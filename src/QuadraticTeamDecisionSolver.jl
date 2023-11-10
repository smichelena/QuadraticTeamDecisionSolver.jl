module QuadraticTeamDecisionSolver

include("quadTeamProblems.jl")
export QuadTeamProblem, checkProblem, residual, GammaNorm, gammaNorm, cost

include("teamMMSE.jl")
export teamMMSEproblem, generateTeamMMSEsamples, sampleComplexNormal

include("fixedPointSolvers.jl")
export jacobiSolver, gaussSeidelSolver

include("kernelMethods.jl")
export exponentialKernel,
	matrixExponentialKernel, gramian, kernelNorm, kernelFunction, kernelRegression
end
