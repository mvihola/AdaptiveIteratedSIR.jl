using CairoMakie, LaTeXStrings, Statistics

#include("../isir_general_state.jl")
import AdaptiveIteratedSIR: _adapt_project

function cm_to_pt(x)
    x/2.54*72
end

function show_timing_and_traj(Ns, Ts, β, out_adapt, Λ=nothing; cms = (16.4, 5), 
    linewidth=0.5, axiswidth=linewidth, old=nothing, line_labels=nothing)
    ξ = out_adapt.ξ
    f = Figure(
        size=cm_to_pt.(cms),
        fontsize=9,
        figure_padding = (0,12,0,0),
        figure_margin = (0,0,0,0)
    )
    function set_default_axis(a)
            a.spinewidth = axiswidth
            a.xtickwidth = axiswidth
            a.ytickwidth = axiswidth
            a.xgridwidth = axiswidth
            a.ygridwidth = axiswidth
#            a.limits = (0,xmax,nothing,nothing)
    end

    #Axis(f[1,2], xscale=log2, title="IREs and loss", yscale=log10),  
    ax = [Axis(f[1,1], title="Estimated cost", xlabel=L"N", ylabel="Run time (s)"),
    Axis(f[1,2], xlabel=L"k", ylabel=L"λ_k", xscale=log10, yscale=log2, title="Adaptation"), 
    Axis(f[1,3], xlabel=L"k", ylabel=L"X_k(1)", title="Trace plot")]
    map(set_default_axis, ax)

    Nmax = maximum(Ns) -1 

    ax[1].xticks = [0, Nmax/2, Nmax]
    ax[1].xtickformat = "{:.0f}"

    #map(a -> hideydecorations!(a; grid=false), ax)

    Makie.scatter!(ax[1], Ns .- 1, Ts, markersize=6, color=:black)
    Makie.ablines!(ax[1], [β[1]], [β[2]], alpha=0.3)
    #xlims = log2.(ax[1].xaxis.attributes.limits[])
    #xs = 2 .^ (range(xlims[1], xlims[2], length=512))
    #predT = β[1] .+ β[2]*xs
    #Makie.lines!(ax[1], xs, predT, alpha=0.3)
    if isnothing(Λ)
        λ = map(xi_k -> begin 
              _, lambda, _  = _adapt_project.(xi_k, Inf)
              lambda
            end, ξ)
        Makie.lines!(ax[2], 1:length(ξ), λ)
    else
        n_ = size(Λ)[2]
        Qs = mapslices(L -> quantile(L, [0.0,0.05,0.25,0.75,0.95,1.0]), Λ, dims=1)
        Makie.band!(ax[2], 1:n_, Qs[1,:], Qs[end,:], color=RGBf(0.85, 0.85, 0.85))
        Makie.band!(ax[2], 1:n_, Qs[2,:], Qs[end-1,:], color=RGBf(0.65, 0.65, 0.65))
        Makie.band!(ax[2], 1:n_, Qs[3,:], Qs[end-2,:], color=RGBf(0.40, 0.40, 0.40))
        #for i = 1:size(Λ)[1]
        #    λ = Λ[i,:]
        #    Makie.lines!(ax[2], n_, λ)
        #end
    end
    n = length(out_adapt.samples)
    thin = ceil(Int, n/10000)
    Makie.lines!(ax[3], 1:thin:n, [x[1] for x in out_adapt.samples[1:thin:n]])
    f, ax
end

function show_results_makie(Ns, IREs, n_cs, IACTs, approx_vs, Λ=nothing; cms = (8.2, 4.5), #(12.7, 6), 
    linewidth=0.5, axiswidth=linewidth, old=nothing, line_labels=nothing,
    Nticks = nticks(Ns, 2))

    Ns_ = Ns .- 1
    if isnothing(old)
        f = Figure(
            size=cm_to_pt.(cms),
            fontsize=9,
            figure_padding = (1,1,1,1),
        )
    end
    function set_default_axis(a)
            a.spinewidth = axiswidth
            a.xtickwidth = axiswidth
            a.ytickwidth = axiswidth
            a.xgridwidth = axiswidth
            a.ygridwidth = axiswidth
            a.xlabel=L"N-1"
            a.xticks = Nticks
#            a.limits = (0,xmax,nothing,nothing)
    end
    n_lines = size(IREs, 1)
    if isnothing(line_labels)
        line_labels = collect(LaTeXString("x_{$(i)}") for i=1:n_lines)
    end

    ax = [Axis(f[1,2], xscale=log2, title="IREs and loss", yscale=log10), Axis(f[1,1], xlabel=L"N", xscale=log2, title="IACT", yscale=log10)]
    map(set_default_axis, ax)
    #map(a -> hideydecorations!(a; grid=false), ax)

    for i = 1:size(IREs, 1)
        Makie.lines!(ax[1], Ns_, IREs[i,:], label=line_labels[i],linewidth=linewidth)
    end
    Makie.scatter!(ax[1], Ns_, n_cs, label="L(N)", markersize=6)

    for i = 1:size(IACTs, 1)
        Makie.lines!(ax[2], Ns_, IACTs[i,:], label=label=line_labels[i], linewidth=linewidth)
    end
    Makie.scatter!(ax[2], Ns_, approx_vs, label="approx.", markersize=6)

    axislegend(ax[2], 
    rowgap=-8, 
    framevisible=false, 
    padding = (0,0,0,0),
    margin = (0,0,0,0))
    f, ax
end

function show_results_makie2(Ns, out_adapt, IREs, n_cs, IACTs, approx_vs; cms = (16.4, 5), Λ = nothing, 
    linewidth=0.5, axiswidth=linewidth, old=nothing, line_labels=nothing, 
    Nticks = nticks(Ns, 2))

    Ns_ = Ns .- 1
    if isnothing(old)
        f = Figure(
            size=cm_to_pt.(cms),
            fontsize=9,
                figure_padding = (1,1,1,1),
        )
    end
    function set_default_axis(a)
            a.spinewidth = axiswidth
            a.xtickwidth = axiswidth
            a.ytickwidth = axiswidth
            a.xgridwidth = axiswidth
            a.ygridwidth = axiswidth
#            a.limits = (0,xmax,nothing,nothing)
    end
    n_lines = size(IREs, 1)
    if isnothing(line_labels)
        line_labels = collect(LaTeXString("x_{$(i)}") for i=1:n_lines)
    end

    ax = [Axis(f[1,3], xscale=log2, title="IREs and loss", yscale=log10,
            xlabel=L"N-1", xticks = Nticks), 
        Axis(f[1,2], xscale=log2, title="IACT", yscale=log10, xlabel=L"N-1", xticks = Nticks),
        Axis(f[1,1],  xlabel=L"k", ylabel=L"λ_k", xscale=log10, yscale=log2, title="Adaptation")]
    map(set_default_axis, ax)
    #map(a -> hideydecorations!(a; grid=false), ax)

    for i = 1:size(IREs, 1)
        Makie.lines!(ax[1], Ns_, IREs[i,:], label=line_labels[i],linewidth=linewidth)
    end
    Makie.scatter!(ax[1], Ns_, n_cs, label="L(N)", markersize=6)

    for i = 1:size(IACTs, 1)
        Makie.lines!(ax[2], Ns_, IACTs[i,:], label=label=line_labels[i], linewidth=linewidth)
    end
    Makie.scatter!(ax[2], Ns_, approx_vs, label="approx.", markersize=6)

    axislegend(ax[2], 
    rowgap=-8, 
    framevisible=false, 
    padding = (0,0,0,0),
    margin = (0,0,0,0))

    if isnothing(Λ)
        λ = map(xi_k -> begin 
              _, lambda, _  = _adapt_project.(xi_k, Inf)
              lambda
            end, ξ)
        Makie.lines!(ax[3], 1:length(ξ), λ)
    else
        n_ = size(Λ)[2]
        Qs = mapslices(L -> quantile(L, [0.0,0.05,0.25,0.75,0.95,1.0]), Λ, dims=1)
        Makie.band!(ax[3], 1:n_, Qs[1,:], Qs[end,:], color=RGBf(0.85, 0.85, 0.85))
        Makie.band!(ax[3], 1:n_, Qs[2,:], Qs[end-1,:], color=RGBf(0.65, 0.65, 0.65))
        Makie.band!(ax[3], 1:n_, Qs[3,:], Qs[end-2,:], color=RGBf(0.40, 0.40, 0.40))
        #for i = 1:size(Λ)[1]
        #    λ = Λ[i,:]
        #    Makie.lines!(ax[2], n_, λ)
        #end
    end

    f, ax
end

f, ax = show_results_makie(Ns_, IREs_, n_cs, IACTs_, approx_iact; 
    line_labels=line_labels, Nticks=Nticks)
#vlines!(ax[1], N_final-1, color=:gray, linewidth=0.5 )
#vlines!(ax[2], N_final-1, color=:gray, linewidth=0.5 )
λ_min = minimum(Λ[:,end])
λ_max = maximum(Λ[:,end])
λ_mean = mean(Λ[:,end])
vspan!(ax[1], λ_min, λ_max, color=:gray)
vspan!(ax[2], λ_min, λ_max, color=:gray)
vlines!(ax[1], λ_mean, color=:gray)
vlines!(ax[2], λ_mean, color=:gray)

f2, ax2 = show_timing_and_traj(Ns, Ts, β, out_adapt, Λ)

f3, ax3 = show_results_makie2(Ns_, out_adapt, IREs_, n_cs, IACTs_, approx_iact; 
    line_labels=line_labels, Nticks=Nticks, Λ=Λ)

#vlines!(ax3[1], N_final-1, color=:gray, linewidth=0.5 )
#vlines!(ax3[2], N_final-1, color=:gray, linewidth=0.5 )
vspan!(ax3[1], λ_min, λ_max, color=:gray)
vspan!(ax3[2], λ_min, λ_max, color=:gray)
vlines!(ax3[1], λ_mean, color=:gray)
vlines!(ax3[2], λ_mean, color=:gray)
