# Generic type to calculate log(mean) of streaming log values recursively
mutable struct RecursiveLogMean{T<:Real}
    log_v::T   # Recursive log(mean) value...
    n::UInt # ..based on n samples
end
# Initialize by first data point:
RecursiveLogMean(log_x::Real) = RecursiveLogMean{typeof(log_x)}(log_x, 1)
# Initialize by type:
RecursiveLogMean(T::DataType) = RecursiveLogMean{T}(convert(T, -Inf), 0)
# ...and by default:
RecursiveLogMean() = RecursiveLogMean(Float64)

function reset!(m::RecursiveLogMean{T}) where {T}
    m.n = 0
    m.log_v = convert(T, -Inf)
    m
end

# Update current value of m.v with new data x
function update!(m::RecursiveLogMean{T}, log_x) where {T}
    m.n += 1
    λ = one(T)/m.n
    if log_x > m.log_v
        max_val = log_x
        x_ = one(T) # =exp(log_x - max_val)
        v_ = exp(m.log_v - max_val)
    else
        max_val = m.log_v
        x_ = exp(log_x - max_val)
        v_ = one(T) # =exp(m.log_v - max_val)
    end
    m.log_v = max_val + log( (one(T)-λ)*v_  + λ*x_ )
    m
end

# Example:
#log_u = -rand(10_000)
#m = RecursiveLogMean()
#for lu in log_u update!(m, lu) end
#m.log_v ≈ log(mean(exp.(log_u))) # (equal mod numerical reasons)