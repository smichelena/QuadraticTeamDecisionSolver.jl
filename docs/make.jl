using Documenter, QuadraticTeamDecisionSolver

makedocs(
    sitename = "QuadraticTeamDecisionSolver",
    pages = [
        "Home" => "index.md",
        "General Utility Functions" => "problemUtils.md",
        "Solvers" => "solvers.md",
        "Kernel Tools" => "kernelTools.md",
    ],
)

deploydocs(
    repo = "github.com/smichelena/QuadraticTeamDecisionSolver.jl",
    versions = nothing,
)
