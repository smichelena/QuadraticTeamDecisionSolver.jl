using QuadraticTeamDecisionSolver, Distributions, LinearAlgebra

struct teamMMSEproblem
	σᵣ::AbstractFloat
	K::Int
	N::Int
	L::Int
	μₕ::Vector{Complex}
	σₕ::Vector{AbstractFloat}
	μₙ::Vector{Complex}
	σₙ::Vector{AbstractFloat}
	ϵ::Vector{AbstractFloat}
end


function sampleComplexNormal(μ::Complex, σ::AbstractFloat, K::Int, N::Int)
	distRe = Distributions.Normal(real(μ), σ / 2)
	distIm = Distributions.Normal(imag(μ), σ / 2)
	return rand(distRe, K, N) + rand(distIm, K, N)im
end

"""
    generateTeamMMSEsamples(t::teamMMSEproblem, m::Int)

generate `m` samples for a team MMSE problem specified by `t`.

# Arguments
- `t::teamMMSEproblem`: a `teamMMSEproblem` object specifying the statistics and sizes of channel and noise measurements 
- `m::Int`: the desired amount of samples

# Returns
- `H`:: A vector of size `t.L` of vectors of size `m` of ``K \\times N`` channel measurement matrices
- `Y`:: A vector of size `t.L` of vectors of size `m` of ``K \\times N \\times L`` agent measurement vectors
- `R`:: A matrix of size `t.L` by `t.L` of vectors of size `m` of ``N \\times N`` matrices
- `r`:: A vector of size `t.L` of vectors of size `m` of ``N \\times N`` matrices

"""
function generateTeamMMSEsamples(t::teamMMSEproblem, m::Int)

	H = [[sampleComplexNormal(t.μₕ[i], t.σₕ[i], t.K, t.N) for _ ∈ 1:m] for i ∈ 1:t.L]
	N = [[sampleComplexNormal(t.μₙ[i], t.σₙ[i], t.K, t.N) for _ ∈ 1:m] for i ∈ 1:t.L]

	Y = Vector{Vector}(undef, t.L)

	for i ∈ 1:t.L
		A = [l == i ? H[l] : (1 - t.ϵ[i]) * H[l] .+ t.ϵ[i] * N[l] for l ∈ 1:t.L]
		Y[i] = [vcat(v...) for v in zip(A...)]
	end

	R = Matrix{Vector}(undef, t.L, t.L)
	for i ∈ 1:t.L
		for j ∈ 1:t.L
			R[i, j] =
				i == j ? (H[i] .|> h -> h') .* H[j] .|> h -> h .+ t.σᵣ : (H[i] .|> h -> h') .* H[j]
		end
	end

	e = zeros(ComplexF64, t.K) # (KxN)'*Kx1 = NxKxKx1 = Nx1
	e[1] = 1.0 + 0.0im # can be later generalized to the k-th entry

	r = Vector{Vector}(undef, t.L)
	for i ∈ 1:t.L
		r[i] = H[i] .|> h -> h'*e
	end

	return H, Y, R, r, e

end


function assembleCoeffSystem(t::teamMMSEproblem, H, F)

    eye = Matrix(I, t.K, t.K)

    res = zeros(ComplexF64, t.K^2, t.K^2)

    for i in 1:t.K
        for j in 1:t.K
            placementMatrix = zeros(ComplexF64, t.K, t.K)
            placementMatrix[i, j] = 1.0 + 0.0im
            pi_i = i != j ? mean(H[j] .* F[j]) : eye
            res += kron(placementMatrix, pi_i)
        end
    end

    return res, -kron(ones(ComplexF64, t.K), eye)

end

function computeOptimum(σ; K = 2, N = 1, L = 2, mc = 10000, mo = 1000, mt = 10000)
	#compute coefficients (pre-train?)
	t = teamMMSEproblem(
		σ,
		K,
		N,
		L,
		zeros(ComplexF64, L), #mu_h
		ones(Float64, L), #sigma_h
		zeros(ComplexF64, L), #mu_n
		0.05 * ones(Float64, L), #sigma_n
		1.0 * ones(Float64, L), # this doesnt make sense for \eps \neq 1
	)
	Hc, Yc, Rc, rc, e1 = generateTeamMMSEsamples(t, mc)
	F = [Rc[i, i] .\ adjoint.(Hc[i]) for i in 1:L]
	M, B = assembleCoeffSystem(t, Hc, F)
	coeffs = M \ B
	C = [coeffs[i:i+L-1, :] for i in 1:L]

	#compute optimal solution (train)
	Ho, Yo, Ro, ro, _ = generateTeamMMSEsamples(t, mo)
	Fo = [Ro[i, i] .\ adjoint.(Ho[i]) for i in 1:L]
	γ = [Fo[i] .|> (f -> f * C[i] * e1) |> x -> vcat(x...) for i in 1:L]
	w_opt = [regression(γ[i], Yo[i]) for i in 1:L]

	#compute optimum (test)
	_, Yt, Rt, rt, _ = generateTeamMMSEsamples(t, mt)
	Ytt = reformatYm(L, mt, Yt)
	Rtt = reformatR(L, mt, Rt)
	rtt = reformatr(L, mt, rt)

	S = [Sample(Y, R, r, 1.0 + 0.0im) for (Y, R, r) in zip(Ytt, Rtt, rtt)]

	return risk(S, [x -> regressor(w_opt[i], Yo[i], x) for i in 1:L])
end