using QuadraticTeamDecisionSolver, Distributions, LinearAlgebra

function sinrExperiment(s, p; samples=10, iterations=10, bandwidth=0.5, regularizer=0.0, eps=0.0)
    t = teamMMSEproblem(s, zeros(Float64, p.N), ones(Float64, p.N), zeros(Float64, p.N), 0.5*ones(Float64, p.N), eps*ones(Float64, p.N));
    Y, R, r = generateTeamMMSEsamples(p, t, samples);
    tR = reformatR(p.N, samples, R);
    tr = reformatr(p.N, samples, r);
    w = [[rand(ComplexF64, samples)] for _ in 1:p.N]
    empiricalJacobiSolver!(p, w, Y, R, r, iterations = iterations, h = bandwidth, λ = regularizer)
    wtest = reformatW(p.N, samples, iterations, w)
    return [urisk(wtest[k], tR, tr) for k in 1:iterations]
end

function bandwidthExperiment(bandwidth, p; samples=10, iterations=10, sinr=1.5, regularizer=1.0, eps=0.0)
    t = teamMMSEproblem(sinr, zeros(Float64, p.N), ones(Float64, p.N), zeros(Float64, p.N), 0.5*ones(Float64, p.N), eps*ones(Float64, p.N));
    Y, R, r = generateTeamMMSEsamples(p, t, samples);
    tR = reformatR(p.N, samples, R);
    tr = reformatr(p.N, samples, r);
    w = [[rand(ComplexF64, samples)] for _ in 1:p.N]
    empiricalJacobiSolver!(p, w, Y, R, r, iterations = iterations, h = bandwidth, λ = regularizer)
    wtest = reformatW(p.N, samples, iterations, w)
    return [urisk(wtest[k], tR, tr) for k in 1:iterations]
end