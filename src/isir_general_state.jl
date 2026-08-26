using Random, ProgressMeter

"""
i = sample_from_categorical(p)

Direct implementation for drawing a random index i with probability p[i].
"""
function _sample_from_categorical(p)
    U = rand(Random.TaskLocalRNG())
    K = 0; S = 0.0; m = length(p)
    while K < m && U >= S
        K = K + 1       # Note that K is not reset!
        @inbounds S = S + p[K] # S is the partial sum up to K
    end
    return K
end

"""
norm_logw!(w, logv)

Normalised probabiliity weights w ∝ exp(logv) with the 'log-sum-trick'
"""
function norm_logw!(w, logv) 
    m = maximum(logv)
    w .= exp.(logv .- m)
    w ./= sum(w)
    nothing
end

# Form i-SIR proposals (to pre-allocated spaces)
# Modifies Z and V (except for at index b)
function _isir_proposal!(Z, V, N, M!, logG, threads, scratch)
    if threads
        #@batch for i = 1:N
        #Threads.@threads :static for i = 2:N
        #    @inbounds z = Z[i]
        #    scr = scratch[Threads.threadid()]
        #    M!(z, scr)
        #    @inbounds V[i] = logG(z, scr)
        #end
        Threads.@threads for i = 2:N
            with_scratch(scratch) do scr
                @inbounds z = Z[i]
                M!(z, Random.TaskLocalRNG(), scr)
                @inbounds V[i] = logG(z, scr)
            end
        end
    else
        for i = 2:N
            @inbounds z = Z[i]
            M!(z, Random.TaskLocalRNG(), scratch)
            @inbounds V[i] = logG(z, scratch)
        end
    end
    nothing
end

# Do i-SIR selection: normalise log-weights & sample
@inline function _isir_selection!(W, V)
    norm_logw!(W, V)
    _sample_from_categorical(W)
end

@inline function _isir_proposal_selection!(Z, W, V, N, M!, logG,  threads, scratch)
    _isir_proposal!(Z, V, N, M!, logG, threads, scratch)
    _isir_selection!(W, V)
end

# Do i-SIR selction with non-integer "number of particles" N-1 < λ ≤ N
# NB: Uses pre-normalised w but only modifies W_
function _isir_fractional_selection(λ, N, W_, w, interpolation)
    if interpolation == :linear
        β = N - λ
        if rand(Random.TaskLocalRNG()) < β # With probability β draw from P_{N-1}:
            #w_ = view(W_, 1:N-1); v_ = view(V, 1:N-1)
            #return _isir_selection!(w_, v_)
            # Only up to N-1 & renormalise:
            w_ = view(W_, 1:N-1); w_ .= w[1:N-1]/(1.0-w[N])
            return _sample_from_categorical(w_)
        else # With probability 1-β draw from P_N
            return _sample_from_categorical(w)
        end
    elseif interpolation == :last
        ### OBSOLETE! ###
        w_ = view(W_, 1:N)
        w_ .= w
        w_[N] *= λ - (N-1) # Modify last weight
        w_ ./= sum(w_)
        return _sample_from_categorical(w_)
    end
end

# Implement projection of ξ so that λ = e^ξ + 1 ≤ Nmax, and determine N
@inline function _adapt_project(ξ, Nmax)
    λ = exp(ξ) + 1
    if λ > Nmax || !isfinite(λ)
        #@warn "Adaptation N > Nmax"
        λ = Nmax; ξ = log(Nmax-1); N = Nmax
    else
        N = Int(ceil(λ))
    end
    ξ, λ, N
end

# Estimate rejection rate and its derivative
function _estimate_rejection_rate_derivative(w, λ, interpolation)
    @assert λ > 1
    N = Int(ceil(λ))
    @assert length(w) >= N
    @inbounds w_1 = w[1]; w_N = w[N]
    # Calculate s = sum of normalized weights except the last:
    s = w_1
    for j = 2:(N-1)
        @inbounds s += w[j]
    end
    if interpolation == :linear
        β = N-λ # kernel (1-β) P_{N-1} + β P_N

        # Direct estimator: omit the last weight
        #ε_λ = w_1*(β/s + (1-β)/(s+w_N))
        #dϵ_λ = w_1*(1/(s+w_N) - 1/s

        # Better estimator: expectation when the 'last' randomly chosen
        p_choose = 1/(N-1) # Each is chosen as the last with this probability
        frac_without_N = (1.0/s)*p_choose # Contribution of the last term
        for j = 2:(N-1)
            # Contribution of the jth term:
            @inbounds frac_without_N += (1.0/(s-w[j]+w_N))*p_choose 
        end
        frac_with_N = 1.0/(s+w_N)
        ε_λ = w_1*(β*frac_without_N + (1-β)*frac_with_N)
        dϵ_λ = w_1*(frac_with_N - frac_without_N)
        return ε_λ, dϵ_λ
        
    elseif interpolation == :last
        ### OBSOLETE! ###
        s += (λ - (N-1))*w_N
        return w_1/s, -w_1*w_N/s^2
    end
end

# Worker for non-adaptive i-SIR
function  _isir_worker(Z, W, V, X, M!, logG, n, thin, N, threads, scratch, elapsedTimes, show_progress)

    progress = Progress(n; enabled=show_progress)
    t0 = time()

    acc = 0; I = 1
    for k = 1:n
        # Set reference to location one:
        @inbounds copyto!(Z[1], Z[I]); V[1] = V[I]

        I = _isir_proposal_selection!(Z, W, V, N, M!, logG, threads, scratch)
        # accept if select non-reference
        acc += (I != 1)

        # Whether we are at 1:thin:n:
        save_k, r = divrem(k-1, thin)
        if r == 0
            sk = save_k + 1
            # save state,
            @inbounds copyto!(X[sk], Z[I])
            # and elapsed time
            @inbounds elapsedTimes[sk] = time() - t0
        end

        next!(progress)
    end
    acc
end

# Worker for adaptive i-SIR
function  _isir_worker_adaptive(Z, W, W_, V, X, M!, logG, n, thin, Nmax, N0, cost, dcost, ξ_, interpolation, threads, scratch, elapsedTimes, show_progress)
    # Acceptance counter & reference index
    acc = 0; I = 1

    progress = Progress(n; enabled=show_progress)
    t0 = time()

    # Initial adaptation state & related λ, N
    ξ = log(N0-1)
    ξ, λ, N = _adapt_project(ξ, Nmax)

    for k = 1:n
        
        # Set reference to one:
        @inbounds copyto!(Z[1], Z[I]); V[1] = V[I]

        # Form proposals
        _isir_proposal!(Z, V, N, M!, logG, threads, scratch)

        # Calculate normalised weights
        w = view(W, 1:N); v = view(V, 1:N)
        norm_logw!(w, v)

        # Draw from the fractional kernel
        I = _isir_fractional_selection(λ, N, W_, w, interpolation)

        # Acceptance if move
        acc += (I != 1)

        # Estimate ε(λ) and ε'(λ)
        ε, dε = _estimate_rejection_rate_derivative(w, λ, interpolation)

        # Gradient like function estimate
        H = dcost(λ)*(1-ε^2) + 2dε*cost(λ)
        
        # Step size
        γ = k^(-0.75)

        # SA step with projection which ensures that λ ≤ Nmax
        ξ, λ, N = _adapt_project(ξ - γ*H, Nmax)

        # Whether we are at 1:thin:n:
        save_k, r = divrem(k-1, thin)
        if r == 0
            sk = save_k + 1
            # Save state,
            @inbounds copyto!(X[sk], Z[I])
            # the adaptation sequence,
            @inbounds ξ_[sk] = ξ
            # and the elapsed time
            @inbounds elapsedTimes[sk] = time() - t0
        end

        # Update progress meter:
        next!(progress)

    end
    acc
end

"""
out = isir(M!, logG, x0, n, N; kwargs)
out = isir(M!, logG, x0, n, N; adaptive=true, kwargs)

Generic iterated SIR.

# Arguments
- `M!`: Function, when called as `M!(x)` simulates a sample to state `x`.
- `logG`: Log-potential function, when called as `logG(x)` calculates logarithmic weight/potential of state `x`
- `n`: Number of iterations
- `N`: Number of proposals (maximum number of proposals when adaptive)

# Keyword arguments
- `adaptive`: Whether adaptive number of proposals is used; default `false`
- `N0`: Initial number of proposals; default `2`
- `cost`: Cost function; default `cost(λ) = 10 + λ`
- `dcost`: Derivative of the cost function; default `1` 
- `interpolation`: Whether fractional kernels are `:linear` interpolated (default) or by down-weighing last (`:last``) (OBSOLETE!)
- `threads`: Whether to parallelise sampling/evaluating the potential
- `show_progress`: Whether to display progress meter; default `false`
- `thin`: Store only every thin'th sample; default 1.

Output `out` is a named tuple with the following fields:
- `samples`: Vector of samples at `1:thin:n`
- `elapsedTimes`: Time (in seconds) from the start of the run at `1:thin:n`
- `ξ`: The adapted parameter at `1:thin:n`
- `acceptance_rate`: Average accept rate.
"""
function isir(M!, logG, x0, n, N; interpolation = :linear,
    adaptive=false, N0=2, cost = λ -> 10 + λ, dcost = λ -> 1, 
    threads=false, newScratch=(()->nothing), show_progress=false, 
    thin=1)

    # Proposals are stored in Z
    Z = [similar(x0) for k=1:N]

    # Storage for log-weights and normalised weights
    V = ones(N); W = ones(N)

    # The first proposal is the reference (initially)
    copy!(Z[1], x0)
    # ...and its log weight
    V[1] = logG(x0, newScratch())
    # Ensure that we start with G(x0)>0:
    @assert V[1] > -Inf

    # Storage for the generated samples
    X = [similar(x0) for k=1:thin:n]

    if interpolation == :last
        @warn  "interpolation=:last is obsolete"
    end
    if threads
        #scratch = [newScratch() for i=1:Threads.maxthreadid()]
        scratch = PerThreadScratch(newScratch)
    else
        scratch = newScratch()
    end
    elapsedTimes = [0.0 for k=1:thin:n]
    if !adaptive
        acc = _isir_worker(Z, W, V, X, M!, logG, n, thin, N, threads, scratch, elapsedTimes, show_progress)
        
        return (samples=X, acceptance_rate=acc/n, elapsedTimes=elapsedTimes)
    else
        # Storage for the adaptation sequence & another scratch space for weights
        ξ = [0.0 for k=1:thin:n]; W_ = ones(N)
        acc = _isir_worker_adaptive(Z, W, W_, V, X, M!, logG, n, thin, N, N0, cost, dcost, ξ, interpolation, threads, scratch, elapsedTimes, show_progress)

        return (samples=X, acceptance_rate=acc/n, elapsedTimes=elapsedTimes, ξ=ξ)
    end
end


