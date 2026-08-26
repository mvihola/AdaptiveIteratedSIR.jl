function isir_timing(M!, logG, x0, Ns, n_timing; newScratch=() -> nothing) 
  # Run once (to avoid timing JIT compilation time)
  isir(M!, logG, x0, 10, Threads.nthreads()+1; adaptive=false, threads=true, newScratch=newScratch)
  Ts = zeros(length(Ns))
  # Run for each N in Ns
  for (i, N) in enumerate(Ns)
    Ts[i] = @elapsed out = isir(M!, logG, x0, n_timing, N; adaptive=false, threads=true, newScratch=newScratch)
  end
  M = [ones(size(Ns)) Ns.-1]  
  # Fit a linear model: Time = offset + factor * N
  β = M \ Ts
  # Define cost based on the above (but normalising so that "factor" = 1)
  cost, dcost = let offset = β[1]/β[2] 
    λ -> offset + λ, λ -> 1
  end
  (Ts=Ts, M=M, β=β, cost=cost, dcost=dcost)
end

