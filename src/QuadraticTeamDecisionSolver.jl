module QuadraticTeamDecisionSolver

include("types.jl")
export QuadTeamProblem, Sample

include("problemUtils.jl")
export checkProblem,
    checkSample,
    checkGamma,
    checkData,
    loss,
    risk,
    splitSampleIntoBlocks,
    splitDataSetIntoBlocks,
    urisk,
    reformatR,
    reformatW,
    reformatr,
    reformatY

include("fixedPointSolvers.jl")
export empiricalAlternatingSolver!, empiricalJacobiSolver!, jacobiPrecodingSolver!, alternatingPrecodingSolver!

include("kernelTools.jl")
export exponentialKernel,
    gramian, kernelFunction, kernelNorm, kernelInterpolation, densityConditionalMean

include("teamMMSEhelpers.jl")
export teamMMSEproblem, generateTeamMMSEsamples, sampleComplexNormal

end
