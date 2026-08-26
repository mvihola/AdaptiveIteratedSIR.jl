# Generic vector-of-scratches struct
struct PerThreadScratch{T}
    data::Vector{T}
    locks::Vector{ReentrantLock}
end

# Constructor with user-supplied function
function PerThreadScratch(creator::Function)
    data  = [creator() for _ in 1:Threads.maxthreadid()]
    locks = [ReentrantLock() for _ in 1:Threads.maxthreadid()]
    return PerThreadScratch(data, locks)
end

# Wrapper which runs function f using per-thread scratch
function with_scratch(f::Function, s::PerThreadScratch)
    tid = Threads.threadid()
    lock(s.locks[tid])
    try
        return f(s.data[tid])
    finally
        unlock(s.locks[tid])
    end
end

# Example usage:
#scratch = PerThreadScratch(() -> zeros(Float64, 10^6))
#Threads.@threads for i in 1:100
#    with_scratch(scratch) do scratch
#        scratch[1:100] .= i
#        println(sum(@view scratch[1:100]))
#    end
#end
