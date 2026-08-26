using Random, LaTeXStrings, CairoMakie, ProgressMeter

Random.seed!(12345)

include("cardoso_model.jl")
include("../tools/estimate_what.jl")

newScratch = make_newScratch(dimension)

using AdaptiveIteratedSIR
import AdaptiveIteratedSIR: _adapt_project
#include("../isir_general_state.jl")

N_max = 2^maxP

# Trial run with these N (for timing)
Ns = 2 .^ (2:maxP) .+ 1
x0 = zeros(dimension)

#include("../timing.jl")
println("Timing:")
timing = isir_timing(M_student!, logG, x0, Ns, n_timing; newScratch=newScratch)

# Run adaptive i-SIR
println("Adaptive i-SIR:")

Λ = zeros(n_adaptive, n_run)
out_adapt = nothing
@showprogress for i = 1:n_adaptive
    global out_adapt = isir(M_student!, logG, x0, n_run, N_max; adaptive=true, threads=true, cost=timing.cost, dcost=timing.dcost, newScratch=newScratch)
    λ = map(ξ -> begin
        _, λ, _  = _adapt_project.(ξ, Inf)
        λ
    end, out_adapt.ξ)
    Λ[i,:] = λ
end

# The final N
_, _, N_final = _adapt_project.(out_adapt.ξ[end], Inf)

Ns_ = sort(vcat(Ns, N_final))

n_runs = Int.(ceil.(target_time ./ ((timing.β[1] .+ timing.β[2]*Ns_)/n_timing)))
outs = Vector{Any}(undef, length(Ns_))
for (i, N) in enumerate(Ns_)
    println("i-SIR N=$(N):")
    outs[i] = isir(M_student!, logG, x0, n_runs[i], N; adaptive=false, threads=true, newScratch=newScratch)
end

include("../tools/isir_empirical_variances.jl")

# Approximate loss:
cs = [approximate_loss(Ns_[i], outs[i].acceptance_rate, timing.cost) for i=1:length(Ns_)]

# Approximate IACTs
approx_iact = [approximate_iact(outs[i].acceptance_rate) for i=1:length(outs)]

# The test function of Cardoso et al.
function test_function(x) 
    # A = [−2, 6] × [−1, 1]^6:
    hit_A = (-2 <= x[1] <= 6)
    for i = 2:7
        hit_A &= (-1 <= x[i] <= 1)
    end
    # B = [0.75, 1.25] × [1, 2] × [−0.1, 0.1]^5:
    hit_B = (0.75 <= x[1] <= 1.25) & (1 <= x[2] <= 2)
    for i=3:7
        hit_B &=  (-0.1 <= x[i] <= 0.1)
    end
    hit_A - hit_B
end

first_element(x) = x[1]

IREs_ = hcat(map(out -> ire_function(out, first_element), outs), 
             map(out -> ire_function(out, test_function), outs))'

IACTs_ = hcat(map(out -> iact_function(out, first_element), outs), 
              map(out -> iact_function(out, test_function), outs))'

# Normalised cost to match minimum of IREs              
n_cs = cs * minimum(IREs_)/minimum(cs)


line_labels = [L"x_1" L"1_A-1_B"]

β = timing.β
Ts = timing.Ts

function nticks(Ns, tick_spacing)
    Ns_ = Ns .- 1
    tick_positions = floor(Int, log2(Ns_[1])):tick_spacing:ceil(Int, log2(Ns_[end]))
    tick_labels = [rich("2", superscript(string(p))) for p in tick_positions]
    #[L"2^{%$p}" for p in tick_positions]
    (2 .^ (tick_positions), tick_labels)
end

est_what = estimate_what(outs, newScratch(), logG)
what = est_what.what