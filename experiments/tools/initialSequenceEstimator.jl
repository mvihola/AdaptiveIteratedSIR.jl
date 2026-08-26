using LinearAlgebra, Statistics

# Convolution of x with itself, at a given lag
function autoconv(x, lag)
    n = length(x)
    @assert lag < n
    x_ = view(x, 1:(n-lag))
    y_ = view(x, (1+lag):n)
    LinearAlgebra.dot(x_, y_)/n
end

# Geyer's (1992) initial sequence estimator (consistent for reversible chains, conservative)
function initial_sequence_estimator(y; max_lag=floor(sqrt(length(y))))
    n = length(y)
    @assert n >= 2

    # Centred series:
    x = y .- Statistics.mean(y)

    pairs_end = Int(min(floor((n-2)/2), ceil(max_lag/2)))

    # Initial estimator:
    γ_0 = autoconv(x, 0)
    γ_1 = autoconv(x, 1)
    v = γ_0 + 2γ_1
    #Γ_prev = γ_0 + γ_1
    for k = 1:pairs_end
        γ_even = autoconv(x, 2k)
        γ_odd = autoconv(x, 2k+1)
        Γ_cur = γ_even + γ_odd
        if Γ_cur <= 0
            break
        else
            v += 2Γ_cur
        end
    end
    v
end

