using Documenter

push!(LOAD_PATH, "../src/")

makedocs(
    sitename = "QuadraticTeamDecisionSolver",
    pages = [
        "Home" => "index.md",
        "General Utility Functions" => "problemUtils.md"
    ],
)

deploydocs(
    repo = "github.com/smichelena/QuadraticTeamDecisionSolver.jl",
    versions = nothing,
)