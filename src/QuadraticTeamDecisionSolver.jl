module QuadraticTeamDecisionSolver

include("types.jl")
export QuadTeamProblem, Sample

include("problemUtils.jl")
export checkProblem, checkSample, checkGamma, checkData, loss, risk, splitSampleIntoBlocks, splitDataSetIntoBlocks

include("fixedPointSolvers.jl")
export naiveIterativeSolver, GeneralOutputMap, generateCrossSamples, assembleSystem, empiricalAlternatingSolver!, empiricalJacobiSolver!

include("kernelTools.jl")
export exponentialKernel, gramian, kernelFunction, kernelNorm, kernelInterpolation, densityConditionalMean

end
