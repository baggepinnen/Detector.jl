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
        Xh = model isa VAE ? autoencode(model,X,false) : autoencode(model,X)
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
    M,U = means(model::VAE, dataset; th=0.5)

Returns two features derived from the mean and uncertainty in the bottleneck of a VAE

#Arguments:
- `th`: Quantile ∈ (0,1) to summarize the feature vector per data point.
"""
function means(model, dataset; th=0.50)
    e = map(dataset) do (X,Y)
        Z = encode(model,X) |> cpu
        m = abs.(Z[:,1,1,:])
        s = Z[:,1,2,:]
        mv = map(eachcol(m)) do x
            quantile(vec(x), th)
        end
        sv = map(eachcol(s)) do x
            quantile(vec(x), th)
        end
        mv,sv
    end
    e = reduce(vcat,e)
    reduce(vcat, first.(e)),reduce(vcat, Base.last.(e))
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




mutable struct Labeler
    model
    features
    givenlabels
    noisylabels
    path::AbstractString
    files::Vector{<:AbstractString}
    lastind::Int
end

function Labeler(path::AbstractString)
    fullpath = joinpath(path, "labelsession")
    isfile(fullpath) || error("Could not find a labelsession in the given path $path")
    deserialize(fullpath)
end

function Labeler(;model, features, path::AbstractString, noisylabels)
    files = joinpath.(path, readdir(path))
    files = filter(x->splitext(x)[end] == ".wav", files)
    givenlabels = Vector{Union{Float64, Missing}}(undef, length(noisylabels))
    fill!(givenlabels, missing)
    Labeler(model, features, givenlabels, noisylabels, path, files, 0)
end


function most_uncertain_missing_label(givenlabels::Set, yh)
    inds = sortperm(abs.(yh.-0.5))
    local ind
    for outer ind in inds
        ind in givenlabels || break
    end
    ind
end

function most_uncertain_missing_label(l::Labeler, yh=innerfit(l))
    inds = sortperm(abs.(yh.-0.5))
    local ind
    for outer ind in inds
        ismissing(l.givenlabels[ind]) && break
    end
    ind, yh[ind]
end


function bootstrap_curve(emodel, truelabels, features, noisylabels)
    truelabels = copy(truelabels)
    features = copy(features)
    noisylabels = copy(noisylabels)
    givenlabels = Set{Int}()
    function fiteval(labels)
        DecisionTree.fit!(emodel,features,Int.(labels))
        yh = DecisionTree.predict_proba(emodel, features)[:,2]
        a = Detector.auc(truelabels,yh,sort(unique(yh)))
        yh,a
    end
    yh,a = fiteval(noisylabels)
    curve = [a]
    Juno.@progress "Bootstrapping" for i in eachindex(truelabels)
        ind = most_uncertain_missing_label(givenlabels, yh)
        push!(givenlabels, ind)
        if noisylabels[ind] == truelabels[ind]
            push!(curve, a)
            continue
        end
        noisylabels[ind] = truelabels[ind]
        yh,a = fiteval(noisylabels)
        push!(curve, a)
    end
    curve
end

function innerfit(l::Labeler)
    DecisionTree.fit!(l.model,l.features,l.noisylabels)
    yh = DecisionTree.predict_proba(l.model, l.features)[:,2]
    yh
end
function start_labeling(l::Labeler)
    yh = innerfit(l)
    for i in eachindex(l.givenlabels)
        ind, prob = most_uncertain_missing_label(l, yh)
        present(l, ind, prob)
        choice = get_input()
        refit = handle_choice(choice, l, ind)
        if refit
            yh = innerfit(l)
        end
    end
end

function present(l::Labeler, ind, prob=nothing)
    if prob === nothing
        @info "Playing file $(l.files[ind])"
    else
        @info "Playing file with probability $(round(prob, digits=3)) $(l.files[ind])"
    end
    wavplay(l.files[ind])
end

function handle_choice(choice, l::Labeler, ind)
    if choice == QUIT
        fullpath = joinpath(l.path, "labelsession")
        serialize(fullpath, l)
        @info "Saved label session to $fullpath"
        error("Aborting (this is not an error, even if it says so)")
    elseif choice == NOSIGNAL
        l.givenlabels[ind] = false
    elseif choice == SIGNAL
        l.givenlabels[ind] = true
    elseif choice == UNDO
        l.givenlabels[l.lastind] = missing
    elseif choice == RETRY
        return false
    elseif choice == STATUS
        printstatus(l)
        return false
    elseif choice isa AbstractFloat
        l.givenlabels[ind] = choice
    end
    l.lastind = ind
    if l.noisylabels[ind] == l.givenlabels[ind]
        return false # nothing to be done
    end
    l.noisylabels[ind] = l.givenlabels[ind]
    return true
end

@enum Choice QUIT=-1 NOSIGNAL SIGNAL UNDO RETRY STATUS

function get_input()
    println("Signal or not? +/-, u to undo, q to quit...")
    answer = readline()
    println("You selected ", answer)
    if answer ∈ ("+","signal","y")
        printstyled("Signal detected\n", color=:green)
        return SIGNAL
    elseif answer ∈ ("-","n")
        printstyled("No signal detected\n", color=:orange)
        return NOSIGNAL
    elseif answer ∈ ("u","z","undo")
        printstyled("No signal detected\n", color=:orange)
        return UNDO
    elseif answer ∈ ("q", "quit", "c", "d")
        printstyled("Quitting\n",color=:red)
        return QUIT
    elseif answer ∈ ("s", "status")
        printstyled("Status\n",color=:blue)
        return STATUS
    elseif tryparse(Float64, answer) != nothing
        val = parse(Float64, answer)
        if 0 <= val <= 1
            printstyled("Uncertain signal detected, probability = $(val)\n", color=:orange)
            return val
        else
            printstyled("Invalid choice\n",color=:red)
        end
    end
    printstyled("Invalid choice\n",color=:red)
    return RETRY
end


function printstatus(l::Labeler)
    println("Number of labels/missing: $(sum(!ismissing, l.givenlabels))/$(sum(ismissing, l.givenlabels))")
    println("Number of positive/negative labels: $(sum(==(1), skipmissing(l.givenlabels)))/$(sum(==(0), skipmissing(l.givenlabels)))")
end

end
