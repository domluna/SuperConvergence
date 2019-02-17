abstract type AbstractCycle end

struct TriangularCycle <: AbstractCycle
    r1::LinRange{Float64}
    r2::LinRange{Float64}
end

TriangularCycle(a::Float64, b::Float64, stepsize::Int) = TriangularCycle(LinRange(a, b, stepsize), LinRange(b, a, stepsize)[2:end])

Base.length(c::TriangularCycle) = length(c.r1) + length(c.r2)
Base.eltype(::Type{TriangularCycle}) = Float64
function Base.iterate(c::TriangularCycle, state=1)
    state > length(c) && (return nothing)
    eta = state <= length(c.r1) ? c.r1[state] : c.r2[state-length(c.r1)]
    #= @info("", state, eta) =#
    return eta, state+1
end

struct ConstantCycle <: AbstractCycle
    val::Float64
    len::Int
end

Base.length(c::ConstantCycle) = c.len
Base.eltype(::Type{ConstantCycle}) = Float64
function Base.iterate(c::ConstantCycle, state=1)
    state > length(c) && (return nothing)
    return c.val, state+1
end
