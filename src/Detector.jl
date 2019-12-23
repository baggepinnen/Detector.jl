module Detector
using Statistics, LinearAlgebra, Serialization
using Base.Threads: @threads, nthreads, threadid, @spawn
using Base.Iterators
using MLDataUtils, Flux, CuArrays, MLBase, Dates, Random, BSON, Plots, DSP, WAV, LazyWAVFiles, Juno, NNlib, SliceMap, DecisionTree
using Zygote
using ImageFiltering # Used in utils

export save_interesting, save_interesting_concat, feature_activations, abs_reconstruction_errors, reconstruction_error, zerocrossing_rate, means


include("mel.jl")
export melspectrogram, mfcc, mel

include("utils.jl")
export pool, bp_filter, rocplot, filterplot, confusplot, to2, to4, flatten, typicalinput, auc

include("data.jl")
export second, mapfiles, serializeall_raw, getexamplewhere, confusing_parts, seconds2hms

include("define_model.jl")
export encode, decode, autoencode, classify, ongpu, maybegpu

include("train.jl")

save_interesting(inds, args...) = save_interesting(findall(inds), args...)

"""
    save_interesting(dataset, inds::Vector{Int}; contextwindow=1)

Save interesting files to disc.

#Arguments:
- `dataset`: you data
- `inds`: the indices of interesting files
- `contextwindow`: how many files to save before and after an interesing file for context.
"""
function save_interesting(dataset, inds::Vector{Int}; contextwindow=1)
    tempdir = mktempdir()
    for ind ∈ inds
        extended_inds = max(1, ind-contextwindow):min(length(dataset), ind+contextwindow)
        sound = map(extended_inds) do i
            deserialize(dataset.files[i])
        end
        sound = reduce(vcat, sound)[:]
        tempfile = joinpath(tempdir, splitpath(dataset.files[ind])[end]*".wav")
        sound .-= mean(Float32.(sound))
        sound .*= 1/maximum(abs.(sound))
        wavwrite(sound, tempfile, Fs=fs)
        println(tempfile)
    end
    save_interesting_concat(dataset, inds, tempdir)
end

function save_interesting_concat(dataset, inds::Vector{Int}, tempdir=mktempdir())
    sound = map(inds) do i
        sound = deserialize(dataset.files[i])
        sound .-= mean(Float32.(sound))
        sound .*= 1/maximum(abs.(sound))
    end
    sound = reduce(vcat, sound)[:]
    tempfile = joinpath(tempdir, "concatenated.wav")
    wavwrite(sound, tempfile, Fs=fs)
    println(tempfile)
end


function feature_activations(model, dataset)
    F = map(dataset) do x
        Z = encode(model, x[:,:,:,:])
        feature_activations = mapslices(norm, Z, dims=(1,2)) |> vec
    end
    reduce(hcat, F)
end

"""
    abs_reconstruction_errors(model, dataset; th=0.7)

Calculate the `th` quantile of absolute reconstruction errors.

#Arguments:
- `th`: Quantile ∈ (0,1)
"""
function abs_reconstruction_errors(model, dataset; th=0.70)
    e = map(dataset) do x
        x = x isa Tuple ? x[2] : x
        X = maybegpu(model, reshape(x,:,1,1,size(x,4)))
        Xh = model isa VAE ? autoencode(model,X,false)[1] : autoencode(model,X)
        ae = abs.(robust_error(X,Xh))
        map(eachcol(ae)) do ae
            quantile(cpu(vec(ae)), th)
        end
    end
    reduce(vcat,e)
end

function reconstruction_error(model, x::AbstractArray{<:Real})
    # CuArrays.reclaim(true)
    X = gpu(x[:,:,:,:])
    Xh = autoencode(model,X)
    robust_error(X,Xh)|> cpu |> vec
end

"""
    M,U,Σ,E,ll = vaefeatures(model::VAE, dataset; th=0.5)

Returns four features derived from the mean and variance in the bottleneck of a VAE, the predicted variance in the output and the reconstruction errors

#Arguments:
- `th`: Quantile ∈ (0,1) to summarize the feature vector per data point.
"""
function vaefeatures(model, dataset; th=0.50)
    e = map(dataset) do (X,Y)
        Z = encode(model,X)
        Zc = Z |> cpu
        m = Zc[:,1,1,:]
        s = exp.(Zc[:,1,2,:])
        Xh,Σ = decode(model, Z)
        mv = map(eachcol(abs.(m))) do x
            quantile(vec(x), th)
        end
        sv = map(eachcol(s)) do x
            quantile(vec(x), th)
        end
        eΣ = exp.(Σ[:,1,1,:])
        sv2 = map(eachcol(cpu(eΣ))) do x
            quantile(vec(x), th)
        end
        E = robust_error(X,Xh)
        ae = abs.(E)
        ae = map(eachcol(ae)) do ae
            quantile(cpu(vec(ae)), th)
        end
        ll = varloss(E,Σ) + kl(Z)
        mv,sv,sv2,ae,cpu(ll)
    end
    e = reduce(vcat,e)
    ntuple(i->reduce(vcat, getindex.(e,i)), length(e[1]))
end

function loglik(m,s)


end

"""
    zerocrossing_rate(x)
    zerocrossing_rate(dataset)

You can probably guess what this does. Accepts vectors, Flux CNN style 4d batched data and iterators of such data.
"""
function zerocrossing_rate(x::AbstractVector)
    mean(abs, DSP.conv(sign.(x), [0.5, -0.5]))
end
zerocrossing_rate(x::Tuple) = zerocrossing_rate(last(x))
zerocrossing_rate(x) = map(zerocrossing_rate, x)

zerocrossing_rate(x::AbstractArray{<:Any, 4}) = vec(mapslices(zerocrossing_rate, x, dims=1))

function zerocrossing_rate(x::AbstractVector{<:AbstractArray{<:Any, 4}})
    r = map(x) do x
        vec(mapslices(zerocrossing_rate, x, dims=1))
    end
    reduce(vcat, r)
end



include("labeler.jl")

end
