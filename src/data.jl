supergc() = GC.gc();GC.gc();GC.gc()

## Some constants for the preprocessing
const fs      = 48_000
const second  = fs
POOL_LENGTH   = 1 # How many frequencies to pool together
POOL_FUNCTION = mean
NSTFT         = round(Int,0.1second) # How many samples to calculate fft on
PASSBAND      = (2e3, 0.49fs)



function filename2starttime(filename, reg=r"[0-9]{6}_[0-9]{6}", df="yymmdd_HHMMSS")
    filename = splitext(splitpath(filename)[end])[1]
    m = String((match(reg, filename)).match)
    d = DateTime(m, df)
    year(d) < 1900 ? d+Year(2000) : d
end

function mapfiles(f, folder, extension=nothing)
    files = readdir(folder)
    extension === nothing || filter!(f->splitext(f)[end] == extension, files)
    map(f, files)
end


process_sound_rfft(sound::Vector{Float32}) = process_sound(rfft, sound)
process_sound_stft(sound::Vector{Float32}) = process_sound(s->stft(s, NSTFT, 0, fs=fs), sound)

const melmat = mel(200000, 20001, 512, 20)
# S is freq × time
# The next function reads raw sound data, and applies the band-pass filter. We then appli STFT to form the spectrogram, and pool frequencies together to reduce the dimension significantly. I have not tuned these parameters much yet, but the idea is to reduce the dimension as much as possible without sacrificing too much performance.
function process_sound(fftfun, sound::Vector{Float32})
    supergc()
    sound = Float32.(vec(sound))
    # sound = Float32.(vec(bp_filter(sound)))
    supergc()
    S = melmat*abs2.(fftfun(sound))
    if POOL_LENGTH > 1
        S = pool(POOL_FUNCTION, S, POOL_LENGTH)
    end
    supergc()
    Float32.(log.(S))
end



#
# using DataFrames
# metafile = DataFrame(load("../Random_megafauna/metadata_extracted.csv"))
# starttime2label = Dict(Pair.(metafile.Starttime, metafile.Speciesabbreviation))


# function serializeall_labels(savepath::AbstractString, df::DistributedWAVFile)
#     sem = Base.Semaphore(2)
#     @sync for file in df.files
#         @async begin
#             filename = splitpath(file.path) |> last |> splitext |> first
#             filestarttime = filename2starttime(filename)
#             Base.acquire(sem)
#             f = Threads.@spawn begin
#                 sound = Float32.(file[:])
#                 supergc()
#                 S = process_sound_stft(sound) .|> Float16
#                 supergc()
#                 nfullsegments = length(file) ÷ 30second
#                 segmentlength = size(S,2) ÷ nfullsegments
#                 for (i,inds) in enumerate(Iterators.partition(1:size(S,2), segmentlength))
#                     starttime = filestarttime + (i-1)Second(30)
#                     label = get(starttime2label, starttime, "unknown")
#                     serialize(joinpath(savepath, string(starttime,"_", label, ".bin")), (S[:,inds],label))
#                 end
#             end
#             fetch(f)
#             supergc()
#             Base.release(sem)
#             println("Done with file ", filename)
#         end
#     end
# end



function serializeall_raw(savepath::AbstractString, df::DistributedWAVFile; segmentlength = 1second)
    @assert segmentlength%second == 0 "No support for segment lengths of fractions of seconds"
    for file in df.files
        filename = file.path |> splitpath |> last |> splitext |> first
        nfullsegments = length(file) ÷ segmentlength
        @info "Starting to read file $filename"
        sound = Float16.(file[:])
        supergc()
        for (i,inds) in enumerate(Iterators.partition(1:length(sound), segmentlength))
            starttime = (i-1)Second(segmentlength÷second)
            serialize(joinpath(savepath, string(filename,"_", starttime, ".bin")),  sound[inds])
        end

        supergc()
        println("Done with file ", filename)
    end
end
