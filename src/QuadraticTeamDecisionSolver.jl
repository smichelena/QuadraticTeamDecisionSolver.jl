module QuadraticTeamDecisionSolver

include("quadTeamProblems.jl")
export QuadTeamProblem, checkProblem, checkGamma, residual, GammaNorm, gammaNorm

include("teamMMSE.jl")
export teamMMSEproblem, generateTeamMMSEsamples, sampleComplexNormal

include("fixedPointSolvers.jl")
export jacobiSolver, gaussSeidelSolver, SORSolver

include("kernelMethods.jl")
export exponentialKernel,
	matrixExponentialKernel, gramian, kernelNorm, kernelFunction, kernelRegression
end
