A package for detecting weird stuff in long acoustic recordings using a sparse autoencoder on raw audio data.
# Installation
This package requires a working Julia GPU environment. Follow instructions at [CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl/)

After that, install this package like this:
```julia
cd(@__DIR__)
using Pkg
Pkg.add("https://github.com/baggepinnen/DiskDataProviders.jl")
Pkg.add("https://github.com/baggepinnen/Detector.jl")
cd("path/to/this/repo")
Pkg.instantiate()
using Detector
```

# Usage examples


## Preprocess data
If time-consuming data loading or preprocessing is required, this package can operate on serialized, preprocessed data files. Serialized files are much faster to read than wav, and storing already preprocessed data cuts down on overhead. To create preprocessed files, we use any of the following

```julia
using Detector, LazyWAVFiles
readpath = "path/to/folder/with/wavfiles"
savepath = "path/to/store/files"
readpath = "/media/fredrikb/storage/crocs/20190821/"
savepath = "/home/fredrikb/arl/crocs_processed/"
df       = DistributedWAVFile(readpath)
second   = 48000
serializeall_raw(savepath, df; segmentlength = 1second)    # Serializes raw audio waveforms, for autoencoding
```


## Create a dataset
For further help using [DiskDataProviders](https://github.com/baggepinnen/DiskDataProviders.jl), see its [documentation]((https://baggepinnen.github.io/DiskDataProviders.jl/latest))
```julia
using DiskDataProviders, MLDataUtils, Flux

function str_by(s)
    m = match(r"(\d+)_(\d+) secon", s)
    parse(Int, m.captures[1])*1000000 + parse(Int, m.captures[2])
end
files = sort(savepath.*mapfiles(identity, savepath, ".bin"), by=str_by)

transform(x) = Flux.normalise(sqrt.(abs.(Float32.(x))).*sign.(x), dims=1) # Some transformation you may want to do on the data
dataset = ChannelDiskDataProvider{Vector{Float32}, Nothing}((1second,), 12, 120, files=files, transform=transform)

t   = start_reading(dataset) # This will start the bufering of the dataset
istaskstarted(t) && !istaskfailed(t) && wait(dataset)
bw  = batchview(dataset) # This can now be used as a normal batchview
x = first(bw)
```

If preprocessing and data loading is cheap and fast, it may be advicable to instead make use of [LengthChannels](https://github.com/baggepinnen/LengthChannels.jl), example:
```julia
using LengthChannels
files = joinpath.(path, readdir(path))
function cpubatches(bs, shuf=false, files=files);
    dataset = LengthChannel{Tuple{Array{Float32,4},Array{Float32,4}}}(length(files)÷bs, 10, spawn=true) do ch
        batch = Array{Float32,4}(undef, inputsize, 1, 1, bs)
        bi = 1
        while true
            for file in (shuf ? shuffle(files) : files)
                sound = transform(wavread(file)[1][:, 1])
                batch[:, 1, 1, bi] .= sound
                bi += 1
                if bi > bs
                    bi = 1
                    bb = copy(batch)
                    put!(ch, (bb, bb))
                end
            end
        end
    end
end
function batches(args...)
    LengthChannel{Tuple{CuArray{Float32,4,Nothing},CuArray{Float32,4,Nothing}}}(cpubatches(args...)) do x
        X = gpu(x[1])
        (X,X)
    end
end
dataset = batches(batchsize)
```

## Train the detector
This package provides a selection of different flavors of autoencoders for use as event detectors. The current options are
- `AutoEncoder`
- `VAE`
Below is and example that trains an `AutoEncoder`. The argument to `AutoEncoder` controls the size of the model, bigger is bigger.

```julia
using Flux, BSON
model = Detector.AutoEncoder(2, sparsify=true)
Detector.encode(model,x) # This will give you the latent channels of x
Detector.train(model, batchview(dataset), epochs=5) # Perform one epoch of training. This will take a long time, a figure will be displayed every now and then. This command can be executed several times
# bson("detector.bson", model=cpu(model)) # Run this if you want to save your trained model
```

To fine tune the detector, you may run a small number of epochs on a particular dataset of interest. Just make sure you apply the same input transformation to this dataset as you did for the training dataset, example:
```julia
sound         = load_your_new_sound()
newdataset    = Vector.(Iterators.partition(sound, 3second))[1:end-1] # remove the last datapoint as this is probably shorter
model         = Detector.load_model()
tunedataset   = dataset.transform.(newdataset)
losses        = Detector.train(model, shuffle(tunedataset), epochs=1)
```

## Detection using reconstruction errors
```julia
using Peaks
model  = Detector.load_model() # Load pre-trained model from disk
errors = abs_reconstruction_errors(model, dataset) # This will take a couple of minutes if done on a large dataset (about half the time of a training epoch)
m,proms = peakprom(errors, Maxima(),1000) # Find peaks in signal
plot(errors);scatter!(m,errors[m], m=(:red, 3), ylabel="Errors", legend=false)
save_interesting(dataset, m, contextwindow=1) # This will save the interesting clips to a folder on disk
```
![window](figs/peaks.png)

The call to `save_interesting` will save all interesting files to disk in wav format for you to listen to. The file paths are printed to `stdout`. A file with all the clips concatenated will also be saved. The `contextwindow` parameter determine how many clips before and after an interesting clip will be saved.

## Detection using VAE latent space
The latent space encoding of the variational autoencoder has proven useful for detection of interesting events. You may use it like this, where the features `M,U` are derived from the mean and uncertainty in the latent-space encoding of the VAE

```julia
using Peaks, MLBase
M,U = Detector.means(model, batches(10))

m,proms = peakprom(-M, Maxima()) # Find peaks in signal
promscoreM = zeros(length(labels))
promscoreM[m] .= proms
rocsM = roc(labels,-M,500)
rocsps = roc(labels,promscoreM,500)
rocplot(rocsM, legend=:bottomright, lab="M auc: $(Detector.auc(rocsM))")
rocplot!(rocsps, legend=:bottomright, lab="M peaks auc: $(Detector.auc(rocsps))")

m,proms = peakprom(U, Maxima()) # Find peaks in signal
promscoreU = zeros(length(labels))
promscoreU[m] .= proms
rocsU = roc(labels,U,500)
rocsps = roc(labels,promscoreU,500)
rocplot!(rocsU, legend=:bottomright, lab="U auc: $(Detector.auc(rocsU))")
rocplot!(rocsps, legend=:bottomright, lab="U peaks auc: $(Detector.auc(rocsps))")
```

![window](figs/roc.svg)

*Note:* detection using peak finding only makes sense if the data is sequential, i.e., samples come from consequtive time windows.
