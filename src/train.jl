function robust_error(X,Xh)
    if size(Xh,1) != size(X,1)
        # @warn "Got unequal sizes"
        m = min(size(Xh,1), size(X,1))
        return Xh[1:m,:,:,:].-X[1:m,:,:,:]
    end
    Xh.-X
end

"""
    train(model, batchview; epochs=10, sparsify=false, α=0.002)

`α` is the stepsize.
"""
function train(model, bw; epochs=10, sparsify, α=0.002, opt = ADAM(α), losses = Float32[], λ=1)
    ps = Flux.params(model)
    Flux.testmode!(model)

    λi = 1e-10

    Juno.@progress "Epochs" for epoch = 1:epochs
        Flux.testmode!(model, false)
        Juno.@progress "Epoch $(epoch)" for (i, x) in enumerate(bw)
            λi = min(λi + λ/2000,λ)
            gs = Flux.gradient(ps) do
                l = loss(model, x)
                push!(losses, Flux.data(l)/var(x))
                yield()
                l
            end
            # i % 3 == 0 && error()
            i % 50 == 0 && GC.gc(); sleep(0.1)
            Flux.Optimise.update!(opt, ps, gs)
            if i % 250 == 0
                # CuArrays.reclaim(true)
                supergc()
                ongpu(model) && CuArrays.BinnedPool.reclaim(true)
                # CuArrays.reclaim(true)
                @async plot(losses, yscale=:log10, legend=false, xlabel="Number of batches", ylabel="Loss") |> display
                GC.gc()
                yield()
                sleep(0.1)
            end
        end
        Flux.testmode!(model, true)

        if epoch % 1  == 0
            opt.eta *= 0.95
            # CuArrays.reclaim(true)
            serialize("$(Dates.now())_$(length(losses)).bin", (cpu(model), opt, losses))
            any(isfinite, losses) && plot(losses, yscale=:log10, legend=false) |> display
            sleep(0.1)
        end
    end

    losses
end
