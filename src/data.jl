supergc() = GC.gc();GC.gc();GC.gc()


"""
    filename2starttime(filename, reg=r"[0-9]{6}_[0-9]{6}", df="yymmdd_HHMMSS")

Tries to figure out what time the sound in a file starts based on the filename

#Arguments:
- `reg`: regex with pattern
- `df`: date format
"""
function filename2starttime(filename, reg=r"[0-9]{6}_[0-9]{6}", df="yymmdd_HHMMSS")
    filename = splitext(splitpath(filename)[end])[1]
    m = String((match(reg, filename)).match)
    d = DateTime(m, df)
    year(d) < 1900 ? d+Year(2000) : d
end

"""
    mapfiles(f, folder, extension=nothing)

Map function `f` over all files in `folder`

#Arguments:
- `extension`: Filter based on a file extension, e.g., ".wav"
"""
function mapfiles(f, folder, extension=nothing)
    files = readdir(folder)
    extension === nothing || filter!(f->splitext(f)[end] == extension, files)
    map(f, files)
end


"""
    serializeall_raw(savepath::AbstractString, df::DistributedWAVFile; segmentlength)

Goes through a distributed wav file and serializes windows of this file to disk for later fast reading.

#Arguments:
- `savepath`: path to folder where files are saved
- `df`: a `DistributedWAVFile` from https://github.com/baggepinnen/LazyWAVFiles.jl
- `segmentlength`: How many samples to save in each file
- `fs`: sample rate
"""
function serializeall_raw(savepath::AbstractString, df::DistributedWAVFile; segmentlength, fs)
    @assert segmentlength%fs == 0 "No support for segment lengths of fractions of seconds"
    for file in df.files
        filename = file.path |> splitpath |> last |> splitext |> first
        nfullsegments = length(file) ÷ segmentlength
        @info "Starting to read file $filename"
        sound = Float16.(file[:])
        supergc()
        for (i,inds) in enumerate(Iterators.partition(1:length(sound), segmentlength))
            starttime = (i-1)Second(segmentlength÷fs)
            serialize(joinpath(savepath, string(filename,"_", starttime, ".bin")),  sound[inds])
        end

        supergc()
        println("Done with file ", filename)
    end
end
