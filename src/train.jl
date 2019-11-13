function robust_error(X,Xh)
        if size(Xh,1) != size(X,1)
            @warn "Got unequal sizes"
            m = min(size(Xh,1), size(X,1))
            return Xh[1:m,:,:,:].-X[1:m,:,:,:]
        end
        Xh.-X
end

"""
    train(model, batchview; epochs=10, sparsify=false, α=0.002)

`α` is the stepsize.
"""
function train(model, bw; epochs=10, sparsify=false, α=0.002)
    ps = Flux.params(model)
    Flux.testmode!(model)

    function loss_(x)
        X = gpu(x)
        Z = encode(model, X, sparsify)
        Xh = decode(model, Z)
        l = sum(abs2.(robust_error(X,Xh))) * 1 // length(X)
        # for layer in model.layers
        #     l += mean(abs, layer.weight)/200
        # end
        # for i = 1:size(Z,4)
        # for ch = 1:size(Z,3)
        #     lreg = sum(abs2,vec(@view(Z[:,:,ch,1])))/k
        #     l += lreg/1000
        #     if lreg > 1e-3
        #         l += sqrt(lreg)/2000
        #     end
        # end
        # end
        # l += 10mean(abs, Z)
        l
    end


    n_params = sum(length, ps)
    opt = gpu(ADAM(α))
    losses = Float32[]

    Juno.@progress "Epochs" for epoch = 1:epochs
        Flux.testmode!(model, false)
        Juno.@progress "Epoch $(epoch)" for (i, x) in enumerate(bw)
            gs = Flux.gradient(ps) do
                l = loss_(x)
                push!(losses, Flux.data(l)/var(x))
                yield()
                l
            end
            Flux.Optimise.update!(opt, ps, gs)
            if i % 500 == 0
                CuArrays.reclaim(true)
                supergc()
                CuArrays.reclaim(true)
                @async plot(losses, yscale=:log10, legend=false, xlabel="Number of batches", ylabel="Loss") |> display
                yield()
            end
        end
        Flux.testmode!(model, true)

        if epoch % 1  == 0
            opt.eta *= 0.95
            CuArrays.reclaim(true)
            serialize("$(Dates.now())_$(length(losses)).bin", (cpu(model), opt, losses))
            any(isfinite, losses) && plot(losses, yscale=:log10, legend=false) |> display
            sleep(0.1)
        end
    end

    losses
end
