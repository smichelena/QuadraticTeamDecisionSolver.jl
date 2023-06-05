module QuadraticTeamDecisionSolver

include("types.jl")
export QuadTeamProblem, Sample

include("problemUtils.jl")
export checkProblem, checkSample, checkGamma, checkData, loss, risk

end
