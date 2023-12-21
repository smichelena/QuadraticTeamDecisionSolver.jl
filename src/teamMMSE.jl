using QuadraticTeamDecisionSolver, Distributions, LinearAlgebra

"""
	struct teamMMSEproblem

A struct to specify parameters for a problem from the team MMSE class.

Attributes:
- P::AbstractFloat: The Signal-to-Interference-plus-Noise Ratio (SINR) for the problem.
- N::Int: The number of transmitters in the problem.
- L::Int: The number of antennas at each transmitter.
- K::Int: The number of receivers in the problem.
- σₕ::Vector{AbstractFloat}: Vector specifying channel covariances.
- σₙ::Vector{AbstractFloat}: Vector specifying noise covariances.
- ϵ::Vector{AbstractFloat}: vector of link coefficients ``\\varepsilon_i \\in (0,1), i \\in \\{1,\\dots,N\\}``.
"""
struct teamMMSEproblem
	P::AbstractFloat #SINR
	N::Int #transmitters
	L::Int #antennas
	K::Int #receivers
	σₕ::Vector{AbstractFloat} #channel covariance
	σₙ::Vector{AbstractFloat} #noise covariance
	ϵ::Vector{AbstractFloat} #link
end

"""
	sampleComplexNormal(σ::AbstractFloat, K::Int, L::Int)

Sample from the circularly symmetric complex Gaussian distribution.

Parameters:
- `σ::AbstractFloat`: The standard deviation of the normal distributions for both the real and imaginary parts of the complex numbers.
- `K::Int`: The number of rows in the output matrix.
- `L::Int`: The number of antennas

Returns:
- A KxN matrix of complex numbers with circularly symmetric Gaussian distribution.
"""
function sampleComplexNormal(σ::AbstractFloat, K::Int, L::Int)
	distRe = Distributions.Normal(0, σ / 2)
	distIm = Distributions.Normal(0, σ / 2)
	return rand(distRe, K, L) + rand(distIm, K, L)im
end

"""
	generateTeamMMSEsamples(t::teamMMSEproblem, m::Int)

generate `m` samples for a team MMSE problem specified by `t`.

# Arguments
- `t::teamMMSEproblem`: a `teamMMSEproblem` object specifying the statistics and sizes of channel and noise measurements 
- `m::Int`: the desired amount of samples

# Returns
- `H`:: A vector of size `t.N` of vectors of size `m` of ``K \\times L`` channel measurement matrices
- `Y`:: A vector of size `t.N` of vectors of size `m` of ``K \\times L \\times N`` agent measurement vectors
- `Q`:: A matrix of size `t.N` by `t.N` of vectors of size `m` of ``L \\times L`` matrices
- `R`:: A vector of size `t.N` of vectors of size `m` of ``L \\times 1`` matrices

"""
function generateTeamMMSEsamples(t::teamMMSEproblem, m::Int)

	H = [[sampleComplexNormal(t.σₕ[i], t.K, t.L) for _ ∈ 1:m] for i ∈ 1:t.N]
	N = [[sampleComplexNormal(t.σₙ[i], t.K, t.L) for _ ∈ 1:m] for i ∈ 1:t.N]

	Y = Vector{Vector}(undef, t.N)

	for i ∈ 1:t.N
		A = [l == i ? H[l] : (1 - t.ϵ[i]) * H[l] .+ t.ϵ[i] * N[l] for l ∈ 1:t.N]
		Y[i] = [vcat(v...) for v in zip(A...)]
	end

	Q = Matrix{Vector}(undef, t.N, t.N)
	for i ∈ 1:t.N
		for j ∈ 1:t.N
			Q[i, j] =
				i == j ? (H[i] .|> h -> h') .* H[j] .|> h -> h .+ 1 / t.P :
				(H[i] .|> h -> h') .* H[j]
		end
	end

	e = zeros(ComplexF64, t.K) # (KxL)'*Kx1 = LxKxKx1 = Lx1
	e[1] = 1.0 + 0.0im # can be later generalized to the k-th entry

	R = Vector{Vector}(undef, t.N)
	for i ∈ 1:t.N
		R[i] = H[i] .|> h -> h' * e
	end

	return Y, Q, -R

end