using Documenter ,QuadraticTeamDecisionSolver

#push!(LOAD_PATH, "../src/")

makedocs(
    sitename = "QuadraticTeamDecisionSolver",
    pages = [
        "Home" => "index.md",
        "General Utility Functions" => "problemUtils.md",
        "Fixed Point Solvers" => "fixedPointSolvers.md",
        "Kernel Tools" => "kernelTools.md"
    ],
)

deploydocs(
    repo = "github.com/smichelena/QuadraticTeamDecisionSolver.jl",
    versions = nothing,
)