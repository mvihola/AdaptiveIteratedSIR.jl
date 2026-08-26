using LinearAlgebra, Optim, ADTypes

# Find the Laplace approximation of the posterior
function find_laplace(log_posterior, x0, targetScratch=nothing, iterations=100_000, g_tol=1e-15, f_tol=0.0)
    nlog_p(x) = -log_posterior(x,targetScratch)
    func = TwiceDifferentiable(nlog_p, x0)
    #o = optimize(func, x0, NelderMead());
    o = optimize(func, x0, BFGS(), Optim.Options(iterations=iterations, g_tol=g_tol, f_abstol=f_tol); autodiff = AutoForwardDiff())
    # The mean
    m = Optim.minimizer(o)
    # The covariance
    Hn = Optim.hessian!(func, m)    
    Hn = Symmetric((Hn+Hn')/2)
    S = inv(Hn)
    # ..and its Cholesky factor
    C = cholesky(S)
    (m=m, C=C)
end

function M_laplace!(x_, rng, s)
    # Set up locals:
    L = s.LA.L
    m = s.LA.m
    z = s.LA.z
    # The actual function:
    randn!(rng, z)
    mul!(x_, L, z)
    x_ .+= m
    nothing
end

function logDensity_laplace(x_, s)
    Li = s.LA.Li
    m = s.LA.m
    dx = s.LA.dx
    z = s.LA.z
    # The actual function:
    dx .= x_ .- m
    mul!(z, Li, dx)
    dx .= z.^2
    s.LA.lognormconst - .5*sum(dx)
end

function M_laplace_safe!(x_, rng, s)
    if rand() <= s.prior.prob
        s.prior.draw!(x_, rng, s.targetScratch)
    else
        M_laplace!(x_, rng, s)
    end
    nothing
end

function build_newScratch(LA, targetScratch=() -> nothing, prior=nothing)
    () -> (
        LA = (m=LA.m, L=LA.C.L, Li=inv(LA.C.L), z=similar(LA.m), dx=similar(LA.m), lognormconst=-.5*log(2*pi) - log(det(LA.C.L))),
        targetScratch = targetScratch(),
        prior=prior
    )
end