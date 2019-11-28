module Detector
using Statistics, LinearAlgebra, Serialization
using Base.Threads: @threads, nthreads, threadid, @spawn
using Base.Iterators
using MLDataUtils, Flux, CuArrays, MLBase, Dates, Random, BSON, Plots, DSP, WAV, LazyWAVFiles, Juno, NNlib, SliceMap
using ImageFiltering # Used in utils

export save_interesting, save_interesting_concat, feature_activations, abs_reconstruction_errors, reconstruction_error


include("mel.jl")
export melspectrogram, mfcc, mel

include("utils.jl")
export pool, bp_filter, rocplot, filterplot, confusplot, to2, to4, flatten, typicalinput

include("data.jl")
export second, mapfiles, serializeall_raw, getexamplewhere, confusing_parts, seconds2hms

include("define_model.jl")
export encode, decode, autoencode, ongpu, maybegpu

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


function feature_activations(model, dataset)
    F = map(dataset) do x
        Z = encode(model, x[:,:,:,:])
        feature_activations = mapslices(norm, Z, dims=(1,2)) |> vec
    end
    reduce(hcat, F)
end


function abs_reconstruction_errors(model, dataset; th=0.90)
    e = map(dataset) do x
        X = gpu(x[:,:,:,:])
        Xh = autoencode(model,X)
        ae = abs.(robust_error(X,Xh)) |> Flux.data
        map(eachcol(ae)) do ae
            quantile(cpu(vec(ae)), th)
        end
        # mean(abs,X - autoencode(model,X,false)) |> Flux.data |> cpu
    end
    reduce(vcat,e)
end

function reconstruction_error(model, x::AbstractArray{<:Real})
    # CuArrays.reclaim(true)
    X = gpu(x[:,:,:,:])
    Xh = autoencode(model,X)
    robust_error(X,Xh) |> Flux.data |> cpu |> vec
end


end
