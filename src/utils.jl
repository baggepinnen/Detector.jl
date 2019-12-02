"μ = 0, σ = 1, operates along dim 1"
function standardize2(x::AbstractMatrix)
    x   = copy(x)
    x .-= mean(x, dims=1)
    x ./= std(x, dims=1)
    x
end

using Printf
summaryprint(l,x,level::Int=0) = (print("\t"^level);@printf("size: %13s, type: %s\n", string(size(x)), string(l)))
summaryprint(x,level::Int=0) = (print("\t"^level);@printf("Input size: %s\n", string(size(x))))


function Base.summary(chain::Chain, x, level=0)
    x = gpu(x)
    summaryprint(x,level)
    for l in chain.layers
        if l isa Chain
            summary(l,x, level+1)
            continue
        end
        x = l(x)
        summaryprint(l,x,level)
    end
end


"""
    bp_filter(x, fs)

Band-pass filter the sound
"""
function bp_filter(x, fs)
    responsetype = Bandpass(PASSBAND...; fs=fs) # NOTE: I reduced upper band stop to get away slightly from Nyquist freq
    designmethod = Butterworth(2)
    filt(digitalfilter(responsetype, designmethod), x)
end



"""
    pool(f,x,s)
Pool the data `x` using Function `f`. Window size `s`
"""
function pool(f,x,s)
    n    = size(x,1) ÷ s
    y    = similar(x, n, size(x,2))
    inds = 1:s
    for i = 1:n
        y[i,:] = f(x[inds,:], dims=1)
        inds   = inds .+ s
    end
    y
end

# @recipe function plot(ch::LowRankModels.ConvergenceHistory)
#     unconstrained = all(iszero, ch.primal_residual) && all(iszero, ch.dual_residual)
#     layout --> (unconstrained ? 1 : 4)
#     # yscale --> :log10
#     xlabel --> "Iteration"
#     s = @series begin
#         title --> "Objective"
#         label --> ""
#         yscale --> :log10
#         ch.objective
#     end
#     if !unconstrained
#         @series begin
#             title --> "Dual Objective"
#             label --> ""
#             ch.dual_objective
#         end
#         @series begin
#             title --> "Primal residual"
#             label --> ""
#             ch.primal_residual
#         end
#         @series begin
#             title --> "Dual residual"
#             label --> ""
#             ch.dual_residual
#         end
#     end
# end
"""
Example
```
using MLBase
rocs = roc(labels,sortperm(errors),200)
rocplot(rocs)
```
"""
rocplot

@userplot Rocplot
@recipe function plot(h::Rocplot)
    rocres = h.args[1]
    fpr,tpr = false_positive_rate.(rocres), true_positive_rate.(rocres)
    auc = 0.
    for xi in 1:length(fpr)-1
        auc += (fpr[xi]-fpr[xi+1])*mean(tpr[xi:xi+1])
    end
    title-->"ROC"
    xlabel-->"False positive rate"
    ylabel-->"True positive rate"
    label-->"AUC: $(round(auc, digits=4))"
    @series fpr,tpr
end


@userplot Filterplot
@recipe function plot(h::Filterplot)
    layer = h.args[1]
    w,n = size(layer.weight)[[2,4]]
    label --> ""
    layout --> n
    seriestype --> (w == 1 ? :line : :heatmap)
    xaxis := false
    yaxis := false
    colorbar --> false
    for i = 1:n
        subplot := i
        @series Flux.data(layer.weight[:,:,1,i])
    end
end

@userplot Confusplot
@recipe function plot(h::Confusplot)
    c = h.args[1]
    seriestype := :heatmap
    ylabel --> "Actual"
    xlabel --> "Predicted"
    yflip -->true
    (c./sum(c,dims=2))[end:-1:1,:]
end



@recipe function plot(c::Chain, X, normalize=true)
    x = X[:,:,:,1:1]
    Flux.testmode!(c, true)
    flattenfun = normalize ? flattenn : flatten
    seriestype := :heatmap
    xaxis --> false
    yaxis --> false
    size --> (1920,1720)
    colorbar --> false
    plotlayers = findall(c.layers) do l
        hasfield(typeof(l), :weight) ||
        l isa DenseNet
    end
    nl = length(plotlayers)
    layout --> nl+1
    @series begin
        title --> "Input"
        dropdims(x,dims=(3,4))
    end
    for i = 1:nl
        @series flattenfun(c[1:plotlayers[i]](x))
    end
    Flux.testmode!(c, false)
end


to2(x) = x[:,:,1,1]
to2f(x) = reshape(x, size(x,1), :)
to4(x) = reshape(x, size(x)[1:2]..., 1, 1)

const It = Iterators
function blockpatch(model, X, npatches=40)
    X = gpu(to4(X))
    Flux.testmode!(model, true)
    bsx = max(size(X,1) ÷ npatches, 1)
    bsy = max(size(X,2) ÷ npatches, 1)
    y0 = model(X) |> data
    mask = similar(X)
    r = map(It.product(It.partition(1:size(X,1), bsx), It.partition(1:size(X,2), bsy))) do (xi,yi)
            mask .= 1
            mask[xi,yi,:,:] .= mean(X[xi,yi,:,:])
            y = model(mask .* X)  |> data
            cpu(dot(y,y0))[]
    end
    r = imfilter(r, Kernel.gaussian(1))
    if size(X,2) > 1
        plot(heatmap(r, colorbar=false, xaxis=false,yaxis=false), heatmap(cpu(X[:,:,1,1])))
    else
        plot(plot(r, title="Inner product"), plot(cpu(X[:,1,1,1]), title="x"))
    end
end

flatten(x) = cpu(data(reshape(x, size(x,1),:)))
function flattenn(x)
    x = cpu(data(x))
    x .-= mean(x, dims=3)
    x ./= (std(x, dims=3) .+ 1e-6)
    flatten(x)
end

function plotgrad(model, X,class=cpu(onecold(densenet(gpu(X)))[]))
    model = removegpu(model)
    Flux.testmode!(model, true)
    g = Flux.gradient(x->model[1:end-1](x)[class], gpu(X))[1] |> data |> cpu
    heatmap(imfilter(abs.(dropdims(g, dims=(3,4))), Kernel.gaussian(2)))
end

function removegpu(model)
    si = findlast(l->l == gpu, model.layers)
    si === nothing || (model = model[si+1:end])
    model
end
# @time plotgrad(datasetvc[1][1][:,:,:,3:3])
# imgg = imfilter(img, Kernel.gaussian(3));

imf(x, s=2) = imfilter(x, Kernel.gaussian(s))
function imf(x::CuArray, s=2)
    k = Kernel.gaussian(s)
    w = mapleaves(data, Conv((9,9), 1=>1, pad=4))
    w.weight[:,:,1,1] .= collect(k)
    gpu(w)(x)
end

function typicalinput(model,Xi,class, iters=500, λ=100)
    Flux.testmode!(model, true)
    Xstart = gpu(reshape(Xi,size(Xi)[1:2]...,1,1))
    X = gpu(Xi[:,:,:,1:1])#gpu(20randn(Float32, size(Xi)[1:2]...,1,1))
    o = Momentum(1)
    # o = ADAM(0.1)
    l = (X)->  - model[1:end-1](X)[class] + λ*mean(abs,X .- Xstart)
    # l = (X)->  - log(model(X)[class] + 1e-9) + λ*mean(abs,X .- Xstart)
    @show losses = [data(l(X))]
    Juno.@progress for i = 1:iters
        # @show i
        gs = Flux.gradient(l, X)[1] |> data # i < 5 ? l : l2
        Flux.Tracker.update!(o,X,gs)
        if i % 5 == 0
            push!(losses, data(l(X)))
        end
        # X .= X - 0.1*gs
    end
    @show l(X)
    X = cpu(data(X[:,:,1,1]))
    Xdiff = X-Xi[:,:,1,1]
    plot(heatmap(imf(Xdiff)), plot(Float64.(losses))) |> display
    # plot(heatmap(X), plot(Float64.(losses))) |> display
    X,losses
end

# pair = class => prediction
function getexamplewhere(model, pair::Pair, dataset; shuffle=false)
    Flux.testmode!(model)
    Y = DiskDataProviders.labels(dataset)
    inds = findall(Y .== pair[1])
    shuffle && (inds = inds[randperm(length(inds))])
    for i in inds
        x = dataset[i][1]
        yh = onecold(model(gpu(to4(x))))[]
        if yh == pair[2]
            return x
        end
    end
    nothing
end
function confusing_parts(model, pair, dataset; kwargs...)
    x = getexamplewhere(model, pair, dataset; kwargs...)
    x === nothing && error("Couldn't find the desired misclassification")
    typicalinput(model, x, pair[1], 500)
end




batchvec(x) = reshape(x, :, size(x, 4))


all_positive(r::AbstractMatrix{<: Real}) = sum(r[2:end,:])
all_negative(r::AbstractMatrix{<: Real}) = sum(r[1,:])

# the number of true positives (r.tp)
MLBase.true_positive(r::AbstractMatrix{<: Real}) = sum(diag(r)[2:end])

# the number of true negatives (r.tn)
MLBase.true_negative(r::AbstractMatrix{<: Real}) = r[1]

# the number of false positives (r.fp)
MLBase.false_positive(r::AbstractMatrix{<: Real}) = all_positive(r) - true_positive(r)

# the number of false negatives (r.fn)
MLBase.false_negative(r::AbstractMatrix{<: Real}) = all_negative(r) - true_negative(r)

# the fraction of positive samples correctly predicted as positive, defined as r.tp / r.p
MLBase.true_positive_rate(r::AbstractMatrix{<: Real}) = true_positive(r) / all_positive(r)

# the fraction of negative samples correctly predicted as negative, defined as r.tn / r.n
MLBase.true_negative_rate(r::AbstractMatrix{<: Real}) = true_negative(r) / all_negative(r)

# the fraction of negative samples incorrectly predicted as positive, defined as r.fp / r.n
MLBase.false_positive_rate(r::AbstractMatrix{<: Real}) = false_positive(r) / all_negative(r)

# the fraction of positive samples incorrectly predicted as negative, defined as r.fn / r.p
MLBase.false_negative_rate(r::AbstractMatrix{<: Real}) = false_negative(r) / all_positive(r)

# Equivalent to true_positive_rate(r::AbstractMatrix{<: Real}).
MLBase.recall(r::AbstractMatrix{<: Real}) = MLBase.true_positive_rate(r)

# the fraction of positive predictions that are correct, defined as r.tp / (r.tp + r.fp).
MLBase.precision(r::AbstractMatrix{<: Real}) = true_positive(r) / (true_positive(r) + false_positive(r))

# the harmonic mean of recall(r::AbstractMatrix{<: Real}) and precision(r::AbstractMatrix{<: Real}).
MLBase.f1score(r::AbstractMatrix{<: Real}) = 2 / (1/precision(r) + 1/recall(r))

function seconds2hms(s)
    h = s÷3600
    s -= 3600h
    m = s÷60
    s -= 60m
    println("$h:$m:$s")
end
