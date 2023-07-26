using QuadraticTeamDecisionSolver, Distributions, LinearAlgebra

function sinrExperiment(
    p::QuadTeamProblem,
    solver!::Function,
    sinr::AbstractFloat;
    samples = 1000,
    iterations = 10,
    bandwidth = 0.5,
    regularizer = 0.0,
    eps = 0.0,
)
    t = teamMMSEproblem(
        sinr,
        zeros(Float64, p.N),
        ones(Float64, p.N),
        zeros(Float64, p.N),
        0.5 * ones(Float64, p.N),
        eps * ones(Float64, p.N),
    )
    Y, R, r = generateTeamMMSEsamples(p, t, samples)
    tR = reformatR(p.N, samples, R)
    tr = reformatr(p.N, samples, r)
    U = [[rand(ComplexF64, samples)] for _ = 1:p.N]
    solver!(
        p,
        U,
        Y,
        R,
        r,
        iterations = iterations,
        h = bandwidth,
        λ = regularizer,
    )
    wtest = reformatU(p.N, samples, iterations, U)
    return [urisk(wtest[k], tR, tr) for k = 1:iterations]
end

function bandwidthExperiment(
    p::QuadTeamProblem,
    solver!::Function,
    bandwidth::AbstractFloat;
    samples = 1000,
    iterations = 10,
    sinr = 1.5,
    regularizer = 0.0,
    eps = 0.0,
)
    t = teamMMSEproblem(
        sinr,
        zeros(Float64, p.N),
        ones(Float64, p.N),
        zeros(Float64, p.N),
        0.5 * ones(Float64, p.N),
        eps * ones(Float64, p.N),
    )
    Y, R, r = generateTeamMMSEsamples(p, t, samples)
    tR = reformatR(p.N, samples, R)
    tr = reformatr(p.N, samples, r)
    w = [[rand(ComplexF64, samples)] for _ = 1:p.N]
    solver!(
        p,
        w,
        Y,
        R,
        r,
        iterations = iterations,
        h = bandwidth,
        λ = regularizer,
    )
    wtest = reformatU(p.N, samples, iterations, w)
    return [urisk(wtest[k], tR, tr) for k = 1:iterations]
end

function samplesExperiment(
    p::QuadTeamProblem,
    solver!::Function,
    samples::Int;
    bandwidth = 0.5,
    iterations = 10,
    sinr = 1.5,
    regularizer = 0.0,
    eps = 0.0,
)
    t = teamMMSEproblem(
        sinr,
        zeros(Float64, p.N),
        ones(Float64, p.N),
        zeros(Float64, p.N),
        0.5 * ones(Float64, p.N),
        eps * ones(Float64, p.N),
    )
    Y, R, r = generateTeamMMSEsamples(p, t, samples)
    tR = reformatR(p.N, samples, R)
    tr = reformatr(p.N, samples, r)
    w = [[rand(ComplexF64, samples)] for _ = 1:p.N]
    solver!(
        p,
        w,
        Y,
        R,
        r,
        iterations = iterations,
        h = bandwidth,
        λ = regularizer,
    )
    wtest = reformatU(p.N, samples, iterations, w)
    return [urisk(wtest[k], tR, tr) for k = 1:iterations]
end
