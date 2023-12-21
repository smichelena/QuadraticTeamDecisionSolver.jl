using QuadraticTeamDecisionSolver
using DocumenterCitations
using Documenter
using Pkg

PROJECT_TOML = Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
PKG_VERSION = PROJECT_TOML["version"]
NAME = PROJECT_TOML["name"]
AUTHORS = join(PROJECT_TOML["authors"], ", ") * " and contributors"
GITHUB = "https://github.com/smichelena/QuadraticTeamDecisionSolver.jl"

bib = CitationBibliography(
	joinpath(@__DIR__, "src", "refs.bib"),
	style = :authoryear,
)

makedocs(
    authors=AUTHORS,
	sitename = "QuadraticTeamDecisionSolver",
    format=Documenter.HTML(
        prettyurls=true,
        canonical="https://smichelena.github.io/QuadraticTeamDecisionSolver.jl/",
        assets=String["assets/citations.css"],
        footer="[$NAME.jl]($GITHUB) v$VERSION docs powered by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl)."
    ),
	pages = [
		"Home" => "index.md",
		"General Static Quadratic Team Problem Functions" => "quadTeamProblems.md",
		"Fixed Point Solvers" => "fixedPointSolvers.md",
		"Scalar and Vector Kernel Methods" => "kernelMethods.md",
		"Team Mininum Mean Square Error Precoding" => "teamMMSE.md",
	],
    plugins=[bib],
)

deploydocs(; repo="github.com/smichelena/QuadraticTeamDecisionSolver.jl")
