A package for detecting weird stuff in long acoustic recordings using a sparse autoencoder on raw audio data.
# Installation
This package requires a working Julia GPU environment. Follow instructions at [CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl/)

After that, install this package like this:
```julia
cd(@__DIR__)
using Pkg
Pkg.add("https://github.com/baggepinnen/DiskDataProviders.jl")
Pkg.add("url/to/this/repo")
cd("url/to/this/repo")
Pkg.instantiate()
using Detector
```

# Usage examples


## Preprocess data

All functionality in this package operates on serialized, preprocessed data files. Serialized files are much faster to read than wav, and storing already preprocessed data cuts down on overhead. To create preprocessed files, we use any of the following

```julia
using Detector, LazyWAVFiles, Dates
readpath = "path/to/folder/with/wavfiles"
savepath = "path/to/store/files"
readpath = "/media/fredrikb/storage/crocs/20190821/"
savepath = "/home/fredrikb/arl/crocs_processed/"
df       = DistributedWAVFile(readpath)
serializeall_raw(savepath, df)    # Serializes raw audio waveforms, for autoencoding
```

## Create a dataset
For further help using [DiskDataProviders](https://github.com/baggepinnen/DiskDataProviders.jl), see its [documentation]((https://baggepinnen.github.io/DiskDataProviders.jl/latest))
```julia
using DiskDataProviders, MLDataUtils, Flux

function str_by(s)
    m = match(r"(\d+)_(\d+) secon", s)
    parse(Int, m.captures[1])*10000 + parse(Int, m.captures[2])
end
files = sort(savepath.*mapfiles(identity, savepath, ".bin"), by=str_by)
labs  = fill(nothing, length(files)) # No labels for autoencoding

second = 48000
transform(x) = Flux.normalise(sqrt.(abs.(reshape(Float32.(x),:,1))).*sign.(x), dims=1) # Some transformation you may want to do on the data
dataset = ChannelDiskDataProvider{Matrix{Float32}, Nothing}((3second,1,1), 2, 120, labels=labs, files=files, transform=transform)

t   = start_reading(dataset) # This will start the bufering of the dataset
istaskstarted(t) && !istaskfailed(t) && wait(dataset)
bw  = batchview(dataset) # This can now be used as a normal batchview
x,y = first(bw)
```

## Train the detector
```julia
using Flux, BSON
model = Detector.model
Detector.encode(model,x) # This will give you the latent channels of x
Detector.train(model, dataset) # Perform one epoch of training. This will take a long time, some figure will be displayed every now and then. This command can be executed several times
# bson("detector.bson", model=cpu(model)) # Run this if you want to save your trained model
```

## Detection using feature activations
```julia
using Peaks
model = Detector.load_model() # Load pre-trained model from disk
errors = reconstruction_errors(model, dataset) # This will take a long time (2-5 minutes) if done on the entire dataset
m,proms = peakprom(errors, Maxima(),100) # Find peaks in signal
plot(errors);scatter!(m,errors[m], m=(:red, 3), ylabel="Errors", legend=false)
save_interesting(dataset, m, contextwindow=1) # This will save the interesting clips to a folder on disk
```
![window](figs/peaks.png)

## Detection using reconstruction errors
```julia
F = feature_activations(model, dataset)
plot(q->quantile(F[:], q), 0, 1)
Fth = F .> quantile(F[:], 0.01)
activation_percentage = mean(Fth,dims=2)[:]
least_activated_featureinds = sortperm(activation_percentage)[:]
least_activated_features = F[least_activated_featureinds, :]
least_activated_features_th = Fth[least_activated_featureinds, :]
plot(least_activated_features_th')
save_interesting(dataset, least_activated_features_th[2,1:10])
```
