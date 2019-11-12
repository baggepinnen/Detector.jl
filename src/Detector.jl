module Detector
using Statistics, LinearAlgebra, Serialization
using Base.Threads: @threads, nthreads, threadid, @spawn
using Base.Iterators
using MLDataUtils, Flux, CuArrays, MLBase, Dates, Random, BSON, Plots, DSP, WAV, LazyWAVFiles, Juno, NNlib
using ImageFiltering # Used in utils

export save_interesting, feature_activations, reconstruction_errors


include("mel.jl")
export melspectrogram, mfcc, mel

include("utils.jl")
export pool, bp_filter, rocplot, filterplot, confusplot, to2, to4, flatten, typicalinput

include("data.jl")
export second, mapfiles, serializeall_raw, getexamplewhere, confusing_parts, seconds2hms

include("define_model.jl")

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
end


function feature_activations(model, dataset; sparsify=false)
    F = map(dataset) do x
        Z = encode(model, x[:,:,:,:], sparsify)
        feature_activations = mapslices(norm, Z, dims=(1,2)) |> vec
    end
    reduce(hcat, F)
end


nf = second÷5
const errorf = gpu(MeanPool((nf,1)))
function reconstruction_errors(model, dataset; sparsify=false)
    map(dataset) do x
        CuArrays.reclaim(true)
        X = gpu(x[:,:,:,:])
        ae = abs.(X - autoencode(model,X,sparsify)) |> Flux.data
        try
            ae = errorf(ae)
        catch
            return 0.
        end
        quantile(cpu(vec(ae)), 0.95)
        # mean(abs,X - autoencode(model,X,false)) |> Flux.data |> cpu
    end
end

function reconstruction_errors(model, x::AbstractArray{<:Real}; sparsify=false)
    CuArrays.reclaim(true)
    X = gpu(x[:,:,:,:])
    ae = abs.(X - autoencode(model,X,sparsify)) |> Flux.data
end


end
