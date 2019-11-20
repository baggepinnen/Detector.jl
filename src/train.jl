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

    function loss_(x)
        X  = gpu(x)
        Z  = encode(model, X, sparsify)
        Xh = decode(model, Z)
        l = sum(abs2.(robust_error(X,Xh))) * 1 // length(X)
        if sparsify && rand() < 0.5
            # l2 = zero(l)
            # for b in 1:size(Z,4), c in 1:size(Z,3)
            #     l2 += sum(gpu([λi]).*norm(@view(Z[:,:,c,b])))
            # end
            # Zn = slicemap(norm, Z, dims=(1,2))
            l += sum(gpu([λi]).*sum(abs,Z)/ size(Z,3))
            # l2 = sum(norm(@view(Z[:,:,c])) c in 1:size(Z,3)*size(Z,4))
            # l += l2 * (1 // size(Z,3))
        end
        l
    end

    Juno.@progress "Epochs" for epoch = 1:epochs
        Flux.testmode!(model, false)
        Juno.@progress "Epoch $(epoch)" for (i, x) in enumerate(bw)
            λi = min(λi + λ/2000,λ)
            gs = Flux.gradient(ps) do
                l = loss_(x)
                push!(losses, Flux.data(l)/var(x))
                yield()
                l
            end
            i % 50 == 0 && GC.gc(); sleep(0.1)
            Flux.Optimise.update!(opt, ps, gs)
            if i % 250 == 0
                # CuArrays.reclaim(true)
                supergc()
                CuArrays.reclaim(true)
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
