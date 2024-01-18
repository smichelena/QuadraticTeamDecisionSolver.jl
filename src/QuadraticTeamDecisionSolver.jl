module QuadraticTeamDecisionSolver

include("quadTeamProblems.jl")
export QuadTeamProblem, checkProblem, checkGamma, residual, GammaNorm, gammaNorm, reformatData

include("teamMMSE.jl")
export teamMMSEproblem, generateTeamMMSEsamples, sampleComplexNormal

include("fixedPointSolvers.jl")
export jacobiSolver, gaussSeidelSolver, SORSolver, solverPreprocessing, optimizedGaussSeidel

include("kernelMethods.jl")
export exponentialKernel,
	matrixExponentialKernel, gramian, covariance, kernelNorm, kernelFunction, kernelRegression
end
