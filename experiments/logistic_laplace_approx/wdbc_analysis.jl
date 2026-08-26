using JLD2

@load "wdbc_out_4threads.jld2"

# Normalised cost to match minimum of IREs              
n_cs = cs * minimum(IREs_)/minimum(cs)

using LaTeXStrings, CairoMakie
line_labels = [L"|x - \mu|", L"\pi(x)"]

function nticks(Ns, tick_spacing)
    Ns_ = Ns .- 1
    tick_positions = floor(Int, log2(Ns_[1])):tick_spacing:ceil(Int, log2(Ns_[end]))
    tick_labels = [rich("2", superscript(string(p))) for p in tick_positions]
    #[L"2^{%$p}" for p in tick_positions]
    (2 .^ (tick_positions), tick_labels)
end
Nticks = nticks(Ns, 1)

include("../tools/plot_code.jl")
Makie.save("wdbcResults.pdf", f3; pt_per_unit=1)

