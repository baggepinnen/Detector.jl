
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
        yh = DecisionTree.predict(emodel, features)
        yh = DecisionTree.predict_proba(emodel, features)[:,2]
        a = Detector.auc(truelabels,yh,sort(unique(yh)))
        yh,a
    end
    yh,a = fiteval(noisylabels)
    a < 0.5 && error("Initial AUC is really bad, this error is to prevent this function from taking forever to run O(N²)")
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
