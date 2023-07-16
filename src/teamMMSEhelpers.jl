using QuadraticTeamDecisionSolver, Distributions, LinearAlgebra

struct teamMMSEproblem
	σᵣ::AbstractFloat
    μₕ::Vector{AbstractFloat}
    σₕ::Vector{AbstractFloat}
    μₙ::Vector{AbstractFloat}
    σₙ::Vector{AbstractFloat}
    ϵ::Vector{AbstractFloat}
end


function sampleComplexNormal(μ, σ, m)
    dist = Distributions.Normal(μ, σ/2)
    return rand(dist, m) + rand(dist,m)im
end

function generateTeamMMSEsamples(p::QuadTeamProblem, t::teamMMSEproblem, m::Int)

	H = [[sampleComplexNormal(t.μₕ[i], t.σₕ[i], p.N) for _ in 1:m] for i in 1:p.N]
	N = [[sampleComplexNormal(t.μₙ[i], t.σₙ[i], p.N) for _ in 1:m] for i in 1:p.N]

	Y = Vector{Vector}(undef, p.N)

    for i in 1:p.N
        A = [l == i ? H[l] : (1 - t.ϵ[i]) * H[l] .+ t.ϵ[i] * N[l] for l in 1:p.N]
    	Y[i] = [vcat(v...) for v in zip(A...)]
    end

	R = Matrix{Vector}(undef, p.N, p.N)
	for i in 1:p.N
		for j in 1:p.N
			R[i,j] = i == j ? (H[i] .|> h -> h').*H[j] .+ t.σᵣ : (H[i] .|> h -> h').*H[j] 

		end
	end

	e = zeros(ComplexF64, p.N)
	e[1] = 1.0 + 0.0im

	r = Vector{Vector}(undef, p.N)
	for i in 1:p.N
		r[i] = (H[i] .|> h -> h').*[e for _ in 1:m] 
	end

	c = 1.0 + 0.0im

	return Y, R, r, c

end
