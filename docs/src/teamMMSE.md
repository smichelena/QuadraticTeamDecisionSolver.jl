# General Types and Utility Functions

## Types
```@docs
QuadTeamProblem{T<:Number}
Sample{T<:Number}
```

## Functions

```@docs
checkProblem(p::QuadTeamProblem)
checkSample(p::QuadTeamProblem, s::Sample)
checkGamma(P::QuadTeamProblem, γ::Vector{<:Function})
checkData(p::QuadTeamProblem, S::Vector{<:Sample})
loss(s::Sample, γ::Vector{<:Function})
risk(S::Vector{<:Sample}, γ::Vector{<:Function})
splitSampleIntoBlocks(p::QuadTeamProblem, s::Sample)
splitDataSetIntoBlocks(p::QuadTeamProblem, S::Vector{<:Sample})
```