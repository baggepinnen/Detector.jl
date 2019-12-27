# # Event detection on simulated dataset
# This notebook performs trainging of the VAE model and illustrates how it can be used to detect interesting events. Since we have simulated the dataset, we have the correct labels available for calculation of AUC scores etc.
cd(@__DIR__)
##
using WAV, LengthChannels, Random, Statistics, LinearAlgebra
using CuArrays
using Flux, BSON, Detector, DSP, AdaptiveFilters
using Base.Iterators
CuArrays.allowscalar(false)
path = "/home/fredrikb/arl/too/RL-W-seq/training/"
files = readdir(path)
labels = occursin.("s", files)
@show mean(labels)
files = path.*files
fs = inputsize = 48000
##


transform(x) = Flux.normalise(sqrt.(abs.(Float32.(x))).*sign.(x), dims=1) # Some transformation you may want to do on the data
invtransform(x) = abs2.(x).*sign.(x)


# ## data loaders
function cpubatches(bs, shuf=false, files=files)
    taskref = Ref{Task}();
    dataset = LengthChannel{Tuple{Array{Float32,4},Array{Float32,4}}}(length(files)÷bs, 10, spawn=true, taskref=taskref) do ch
        batch = Array{Float32,4}(undef, inputsize, 1, 2, bs)
        ## ybatch = Array{Float32,4}(undef, inputsize, 1, 1, bs)
        bi = 1
        while true
            for file in (shuf ? shuffle(files) : files)
                sound = wavread(file)[1][:, 1]
                sound = bp_filter(sound, (200/48000, 0.40))
                soundf = sound .- adaptive_filter(sound)
                sound = transform(sound)
                soundf = transform(soundf)
                ## ybatch[:, 1, 1, bi] .= sound
                length(sound) == 48000 || @warn file
                batch[:, 1, 1, bi] .= sound # augment(sound, pip)
                batch[:, 1, 2, bi] .= soundf # augment(sound, pip)

                bi += 1
                if bi > bs
                    bi = 1
                    bb = copy(batch)
                    put!(ch, (bb, bb))
                end
            end
        end
    end, taskref
end
function batches(args...) # This function puts the data on the GPU
    c, tr = cpubatches(args...)
    LengthChannel{Tuple{CuArray{Float32,4,Nothing},CuArray{Float32,4,Nothing}}}(c) do x
        X = gpu(x[1])
        (X,X)
    end, tr
end

dataset,tr = batches(4, true)
@show length(dataset)
x,y = first(dataset)
model = Detector.VAE2(20, c0=1e-9, ci=1/1e4) |> gpu

function callback()
    f1 = current()
    x,y = first(dataset)
    Xh, Σ = autoencode(model, x, false)
    i = rand(1:(48000-600))
    fig = Plots.plot(cpu(Xh[i:i+600,1,1,1]), ribbon = cpu(exp.(Σ[i:i+600,1,1,1]./2)), show=false)
    Plots.plot!(fig, cpu(x[i:i+600,1,1,1]), show=false)
    Plots.plot(f1, fig) |> display
end

# ## Train the model
# We train the model using 800 batches only. This is not even a full epoch, but it's enough for the model to learn the background distribution well without learning the signals.
opt = AMSGrad(0.0001)
losses = Tuple{Float32,Float32}[]
GC.gc(true);GC.gc();GC.gc(true);GC.gc();GC.gc();CuArrays.reclaim()/1e6
Detector.train(model, take(dataset,800), epochs=1, opt=opt, losses=losses, plotinterval=200, saveinterval=1, layout=2, yscale=[:log10 :identity], cb=callback)


# ## Feature calculation
# Below we calculate some features derived from the trained model, as well as standard features like zero-crossing rate.

@time cdataset = collect(cpubatches(1)[1])
ch1(x) = x[:,:,1:1,:]
@time M,U,Σ,errors,ll = Detector.vaefeatures(model, batches(10)[1], th=0.9)
@time zcrv = reduce(vcat, map(zerocrossing_rate ∘ ch1 ∘ first, cdataset))

# We can see how well each feature work for binary classification by itself. No feature is perfect, but the model reconstruction error should give a large tpr with very small fpr, which is crucial for the bootstrapping procedure that comes next.
gr()
closeall()
using Peaks, MLBase
m,proms = peakprom(M, Maxima()) # Find peaks in signal
promscoreM = zeros(length(labels))
promscoreM[m] .= proms
rocsM = MLBase.roc(labels,M,sort(unique(M)))
rocsps = MLBase.roc(labels,promscoreM,sort(unique(promscoreM)))
rocplot(rocsM, legend=:bottomright, lab="M auc: $(Detector.auc(rocsM))")
## rocplot!(rocsps, legend=:bottomright, lab="Mp auc: $(Detector.auc(rocsps))")

m,proms = peakprom(-U, Maxima()) # Find peaks in signal
promscoreU = zeros(length(labels))
promscoreU[m] .= proms
rocsU = MLBase.roc(labels,-U,sort(unique(-U)))
rocsps = MLBase.roc(labels,promscoreU,sort(unique(promscoreU)))
rocplot!(rocsU, legend=:bottomright, lab="U auc: $(Detector.auc(rocsU))")


m,proms = peakprom(Σ, Maxima()) # Find peaks in signal
promscoreΣ = zeros(length(labels))
promscoreΣ[m] .= proms
rocsΣ = MLBase.roc(labels,Σ,sort(unique(Σ)))
rocsΣp = MLBase.roc(labels,promscoreΣ,sort(unique(promscoreΣ)))
rocplot!(rocsΣ, legend=:bottomright, lab="\\Sigma auc: $(Detector.auc(rocsΣ))")
rocplot!(rocsΣp, legend=:bottomright, lab="\\Sigmap auc: $(Detector.auc(rocsΣ))")

rocs = MLBase.roc(labels,1 ./zcrv,sort(unique(1 ./zcrv)))
rocplot!(rocs, legend=:bottomright, lab="izcr auc: $(Detector.auc(rocs))")

m,proms = peakprom(errors, Maxima()) # Find peaks in signal
promscoree = zeros(length(labels))
promscoree[m] .= proms
rocs = MLBase.roc(labels,errors,sort(unique(errors)))
rocsps = MLBase.roc(labels,promscoree,sort(unique(promscoree)))
rocplot!(rocs, lab="rec.err. auc: $(Detector.auc(rocs))")
## rocplot!(rocsps, lab="rec.err.p auc: $(Detector.auc(rocs))")


# ## Boostrapped supervised learning
# In this step, we derive noisy labels based on one of the features calculated above. These labels will not be accurate, but they will allow us to train an initial classifier. It's important to choose the classification threshold for the initial classifier very conservatively so as to favor false negatives rather than false positives.
histogram(errors)
# The histogram should give a good sense of a resonable threshold
noisylabels = (errors .> 0.77)[1:end]
## noisylabels[noisylabels .== 0] .= 0.01
truelabels = labels[1:end]
features = Xfull[1:end,:]

# Next we define a classifier. I use a simple random forest because it gives a probabalistic output and is *very* fast to train.
using DecisionTree
using MonteCarloMeasurements

emodel = RandomForestClassifier(n_trees=40, n_subfeatures=5)
DecisionTree.fit!(emodel,features,Int.(noisylabels))
## yh = DecisionTree.predict(emodel, features)
yh = DecisionTree.predict_proba(emodel, features)[:,2]
# We can compare the initial classifier to the feature that was used to derive it's noisy labels. Sometimes this classifier already does better than it's supposed to based on the feature used, but not always. It should at least not do much worse.
@show Detector.auc(labels,yh)
rocplot(MLBase.roc(labels,yh,sort(unique(yh))))
rocplot!(MLBase.roc(labels,errors,sort(unique(errors))))

# Below, we calculate what we call the *boostrap curve*. It shows the AUC of the classifier as a function of the percentage of true labels given. At 0 percent, it only uses the heuristically derived noisy labels from above. Using this classifier, all datapoints are classified, and the model will query the oracle for the label of the datapoint with a posterior probability closest to 50%. If the true label for this datapoint turned out to be the same as the previously used noisy label, nothing is done, if the tru label differed, the model is retrained.
curves = tmap(1:6) do i
    Detector.bootstrap_curve(emodel, truelabels, features, noisylabels)
end
pcurves = MvParticles(curves)
Plots.plot(LinRange(0, 100, length(curves[1])), pcurves, 0, ylabel="AUC", xlabel="Percent of true labels given", legend=false, ylims=(0,1))

# This process can be done without the true labels available as well, in which case the user will be queried for the label in the terminal
labeler = Detector.Labeler(model=emodel, features=features, path=path, noisylabels=Float64.(noisylabels))
Detector.start_labeling(labeler)
Detector.printstatus(labeler)
confusmat(2,1 .+ labels, 1 .+ round.(Int, labeler.noisylabels))

giveninds = .!ismissing.(labeler.givenlabels)
confusmat(2,1 .+ labels[giveninds], 1 .+ round.(Int, labeler.noisylabels[giveninds]))


# # Analysis of the dataset

# ## Embeddings
# Below we calculate two kinds of embeddings, the first one as Welch spectra
using DSP, AudioClustering
Ps = welch_pgram.(vec.(first.(cdataset)), 1024)
powers = getfield.(Ps, :power)
P = reduce(hcat, powers)
Plots.scatter(threeD(v1(P,1)'), group=labels, size=(1920,1080))


# The second embedding is derived from the poles of rational models
using DSP, AudioClustering, SpectralDistances
linearmodels = tmap(cdataset) do d
    fitmodel(TLS(na=20), vec(first(d)))
end
embs = embeddings(linearmodels)
Plots.scatter(threeD(v1(embs,1)'), group=labels, size=(1920,1080))

# We can run some topological dimensionality reduction on these embeddings to visualize them better, we use UMAP below
using UMAP
uemb = umap(v1(embs,1), 2, n_neighbors=5)
Plots.scatter(eachrow(uemb)..., group=labels, size=(1920,1080), alpha=0.8)
#
uembP = umap(v1(P,1), 2, n_neighbors=5)
Plots.scatter(eachrow(uembP)..., group=labels, size=(1920,1080), alpha=0.8)
# The commands below gives you a scatter plot where each point is clickable to start playing the sound it represents
AudioClustering.interactive_scatter(uemb[1,:],uemb[2,:], files, markersize=0.2labels .+ 0.1, color=labels)
AudioClustering.interactive_scatter(uembP[1,:],uembP[2,:], files, markersize=0.2labels .+ 0.1, color=labels)
# Literate.notebook("simulated_data.jl", ".", config=Dict("documenter"=>false, "execute"=>false)) #src
