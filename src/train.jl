function robust_error(X,Xh)
    if size(Xh,1) != size(X,1)
        # @warn "Got unequal sizes"
        m = min(size(Xh,1), size(X,1))
        return Xh[1:m,:,:,:].-X[1:m,:,:,:]
    end
    Xh.-X
end

Base.Matrix(x::Vector{<:Tuple}) = reduce(hcat, getindex.(x,i) for i in eachindex(x[1]))
Base.Matrix(x::Vector) = x[:,:]

"""
    train(model, batchview; epochs=10, sparsify=false, α=0.002)

`α` is the stepsize.
"""
function train(model, bw; epochs=10, sparsify, α=0.002, opt = ADAM(α), losses = Float32[], λ=1, kwargs...)
    ps = Flux.params(model)
    Flux.testmode!(model)

    λi = 1e-10

    Juno.@progress "Epochs" for epoch = 1:epochs
        Flux.testmode!(model, false)
        Juno.@progress "Epoch $(epoch)" for (i, x) in enumerate(bw)
            λi = Float32(min(λi + λ/2000,λ))
            gs = Flux.gradient(ps) do
                l = loss(model, x)
                if l isa Tuple
                    l1,l2,l3 = l
                    push!(losses, Flux.data.((l1/var(x),l2,l3)))
                    return l1+l2+l3
                else
                    push!(losses, Flux.data(l/var(x)))
                    return l
                end
            end
            # i % 3 == 0 && error()
            # i % 100 == 0 && GC.gc();
            Flux.Optimise.update!(opt, ps, gs)
            if i % 500 == 0
                # CuArrays.reclaim(true)
                supergc()
                ongpu(model) && CuArrays.BinnedPool.reclaim(true)
                # CuArrays.reclaim(true)
                plot(Matrix(losses); legend=false, xlabel="Number of batches", kwargs...) |> display
            end
        end
        Flux.testmode!(model, true)

        if epoch % 1  == 0
            opt.eta *= 0.95
            # CuArrays.reclaim(true)
            serialize("$(Dates.now())_$(length(losses)).bin", (cpu(model), opt, losses))
            plot(Matrix(losses), legend=false) |> display
            sleep(0.1)
        end
    end

    losses
end
