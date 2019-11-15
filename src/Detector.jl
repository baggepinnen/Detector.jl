module Detector
using Statistics, LinearAlgebra, Serialization
using Base.Threads: @threads, nthreads, threadid, @spawn
using Base.Iterators
using MLDataUtils, Flux, CuArrays, MLBase, Dates, Random, BSON, Plots, DSP, WAV, LazyWAVFiles, Juno, NNlib
using ImageFiltering # Used in utils

export save_interesting, save_interesting_concat, feature_activations, reconstruction_errors


include("mel.jl")
export melspectrogram, mfcc, mel

include("utils.jl")
export pool, bp_filter, rocplot, filterplot, confusplot, to2, to4, flatten, typicalinput

include("data.jl")
export second, mapfiles, serializeall_raw, getexamplewhere, confusing_parts, seconds2hms

include("define_model.jl")
export encode, decode, autoencode

include("train.jl")


function load_model()
    global model = BSON.load(joinpath(@__DIR__(),"../detector.bson"))[:model] |> gpu
end



save_interesting(inds, args...) = save_interesting(findall(inds), args...)
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


function feature_activations(model, dataset; sparsify=true)
    F = map(dataset) do x
        Z = encode(model, x[:,:,:,:], sparsify)
        feature_activations = mapslices(norm, Z, dims=(1,2)) |> vec
    end
    reduce(hcat, F)
end


nf = second÷5
const errorf = gpu(MeanPool((nf,1)))
function reconstruction_errors(model, dataset; sparsify=true)
    map(dataset) do x
        CuArrays.reclaim(true)
        X = gpu(x[:,:,:,:])
        Xh = autoencode(model,X,sparsify)
        ae = abs.(robust_error(X,Xh)) |> Flux.data
        try
            ae = errorf(ae)
        catch
            return 0.
        end
        quantile(cpu(vec(ae)), 0.90)
        # mean(abs,X - autoencode(model,X,false)) |> Flux.data |> cpu
    end
end

function reconstruction_errors(model, x::AbstractArray{<:Real}; sparsify=true)
    CuArrays.reclaim(true)
    X = gpu(x[:,:,:,:])
    ae = abs.(X - autoencode(model,X,sparsify)) |> Flux.data |> cpu
end


end
