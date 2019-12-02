function robust_error(X,Xh)
    # @show size(X)
    if ndims(X) > 2
        X = dropdims(X,dims=(2,3))
    end
    if ndims(Xh) > 2
        Xh = dropdims(Xh,dims=(2,3))
    end
    if size(Xh,1) != size(X,1)
        # @warn "Got unequal sizes"
        m = min(size(Xh,1), size(X,1))
        return Xh[1:m,:].-X[1:m,:]
    end
    Xh.-X
end

Base.Matrix(x::Vector{<:Tuple}) = reduce(hcat, getindex.(x,i) for i in eachindex(x[1]))
Base.Matrix(x::Vector) = x[:,:]

lossvec(::Any) = Float32[]
lossvec(model::MixtureAutoencoder) = Tuple{Float32,Float32,Float32}[]

Zygote.@nograd pushlog!(losses, x) = push!(losses,x)

"""
    train(model, batchview; epochs=10, α=0.002)

`α` is the stepsize.
"""
function train(model, dataset; epochs=10, α=0.002, opt = ADAM(α), losses = lossvec(model), plotinterval=length(dataset)÷2, saveinterval=max(epochs÷2,1), kwargs...)
    ps = Flux.params(model)
    # Flux.testmode!(model)


    Juno.@progress "Epochs" for epoch = 1:epochs
        # Flux.testmode!(model, false)
        Juno.@progress "Epoch $(epoch)" for (i, x) in enumerate(dataset)
            gs = Flux.gradient(ps) do
                l = loss(model, x, losses)
                l isa Tuple ? +(l...) : l
            end
            Flux.Optimise.update!(opt, ps, gs)
            if i % plotinterval == 0
                GC.gc()
                plot(Matrix(losses); legend=false, xlabel="Number of batches", kwargs...) |> display
            end
        end
        # Flux.testmode!(model, true)

        if epoch % saveinterval  == 0
            serialize("$(Dates.now())_$(length(losses)).bin", (cpu(model), opt, losses))
            plot(Matrix(losses); legend=false, xlabel="Number of batches", kwargs...) |> display
        end
    end

    losses
end
