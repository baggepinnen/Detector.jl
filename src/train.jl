function train(model, dataset)
    bw      = batchview(dataset)
    pars = Flux.params(model)
    Flux.testmode!(model)

    function loss(x)
        X = gpu(x)
        Z = encode(model, X, false)
        Xh = decode(model, Z)
        l = sum(abs2.(Xh.-X)) * 1 // length(X)
        for layer in model.layers
            l += mean(abs, layer.weight)/20
        end
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

    function ls_loss(x)
        X = gpu(x)
        Z = encode(model, X, false)
        Xh = decode(model, Z)
        l = ls_loss(X,Xh)
        # for layer in model.layers
        #     l += mean(abs, layer.weight)/20
        # end
        l
    end
    function loss2(x)
        X = gpu(x)
        Z = encode(model, X, false)
        Xh = decode(model, Z)
        sum(abs.(Xh.-X)) * 1 // length(X)
    end
    n_params = sum(length, pars)
    opt = gpu(ADAM(0.002))
    opt2 = gpu(ADAM(0.002))
    losses = Float32[]

    ps = Flux.params(model)
    for i = 1:10
        Flux.testmode!(model, false)
        L = 0f0
        Juno.@progress for (i, (x,_)) in enumerate(bw)
            gs = Flux.gradient(ps) do
                # @show l = ls_loss(x)
                l = loss(x)
                push!(losses, Flux.data(l))
                yield()
                # L += data(l)
                l
            end
            Flux.Optimise.update!(opt, ps, gs)
            if i % 500 == 0
                CuArrays.reclaim(true)
                supergc()
                CuArrays.reclaim(true)
                plot(losses, yscale=:log10, legend=false, xlabel="Number of batches", ylabel="Loss") |> display
            end
        end
        # push!(losses, L/length(bw))
        Flux.testmode!(model, true)

        if i % 1  == 0
            opt.eta *= 0.95
            CuArrays.reclaim(true)
            serialize("$(Dates.now())_$(length(losses)).bin", (cpu(model), opt, losses))
            any(isfinite, losses) && plot(losses, yscale=:log10, legend=false) |> display
            sleep(0.1)
        end
    end

    losses
end
