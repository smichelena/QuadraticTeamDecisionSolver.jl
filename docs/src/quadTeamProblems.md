# General Types and Utility Functions

These types and functions provide basic tools for defining and dealing with static quadratic team problems.

## Types

```@docs
QuadTeamProblem{T<:Number}
```

## Functions

```@docs
checkProblem(p::QuadTeamProblem)
checkGamma(p::QuadTeamProblem, γ::Vector{<:Function})
residual(
  m::Int,
  p::QuadTeamProblem,
  kernels::Vector{<:Function},
  γ::Vector{<:Vector{<:Vector}},
  Y::Vector{<:Vector},
  Q::Matrix{<:Vector},
  R::Vector{<:Vector},
  λ::Vector{<:AbstractFloat},
 )
gammaNorm(f::Function, Y::AbstractVector)
GammaNorm(F::Vector{<:Function}, Y::AbstractVector)
```
