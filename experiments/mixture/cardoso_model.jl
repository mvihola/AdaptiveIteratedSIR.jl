# Adapted from:
# https://proceedings.neurips.cc/paper_files/paper/2022/hash/04bd683d5428d91c5fbb5a7d2c27064d-Abstract-Conference.html

using Distributions

# Proposal: t_ν(0,I)
function M_student!(x_, rng, scratch, ν=3)
    d = length(x_)
    u = rand(rng, Chisq(ν))
    u = (u > 0) ? u : 1.0 # Make sure there's no division by 0
    s = sqrt(ν/u)
    @inbounds for i = 1:d
        x_[i] = s * randn(rng)
    end
    nothing
end

function make_newScratch(d::Int; p::Float64=3.0)
    function newScratch()
        (d=d, p=p, m1 = ones(d), m2 = vcat(-2.0, zeros(d-1)))
    end
end

function log_target(x, scr)
        L1 = 0.0
        L2 = 0.0
        @inbounds for i in 1:scr.d
            L1 -= 0.5scr.p * (x[i] - scr.m1[i])^2
            L2 -= 0.5scr.p * (x[i] - scr.m2[i])^2
        end
        Lmax = max(L1, L2)
        L1 -= Lmax
        L2 -= Lmax
        return Lmax + log(exp(L1) + exp(L2))
end

# t_ν(0,I) log density (up to a constant)
function log_d_student(x_, ν=3)
    d = length(x_)
    s2 = mapreduce(x -> x^2, +,  x_)
    -(d+ν)/2*log(1 + s2/ν)
end

# Importance weight
function logG(x_, scratch)
    log_target(x_, scratch) - log_d_student(x_)
end
