
# using Flux: data
using Zygote
# data(x) = x
const k = 15

function sparsify_wta!(Zc)
    @inbounds for bi in 1:size(Zc,4)
        for ci in 1:size(Zc,3)
            a = argmax(abs.(@view(Zc[:,:,ci,bi]))[:])
            Zc[:,:,ci,bi] .= 0.01 .* randn.()
            Zc[a,:,ci,bi] .= 1
        end
    end
    # sum(Zc, dims=1:3)
end

function sparsify_channels!(Zc)
    @inbounds for bi in 1:size(Zc,4)
        i = 0
        am = 0.
        for ci in 1:size(Zc,3)
            a = sum(abs, @view(Zc[:,:,ci,bi]))
            if a > am
                am = a
                i = ci
            end
        end
        rand() < 0.01 && @show i
        # acts = vec(sum(abs, @view(Zc[:,:,:,bi]), dims=1))
        # i = argmax(acts)
        Zc[:,:,:,bi] .= 0.1randn(size(Zc)[1:3]...)
        Zc[:,:,i,bi] .= 1
    end
end

Zygote.@nograd sparsify_wta!

function oneactive(Z)
    Zc = cpu(Zygote.dropgrad(Z))
    sparsify_wta!(Zc)
    Z = gpu(Zc) .* Z
end
encode(model, X::Array, sparsify=true) = cpu(data(encode(model, gpu(X), sparsify)))
encoderlength(model) = length(model) ÷ 2
# encoderlength(model) = length(model)-2
function encode(model, X, sparsify=true)
    X = reshape(X, size(X,1), 1, 1, :)
    sparsify ? oneactive(model[1:encoderlength(model)](X)) : model[1:encoderlength(model)](X)
end

decode(model, Z) = model[encoderlength(model)+1:end](Z)
autoencode(mode,x, sparsify=true) = decode(model,encode(model, x, sparsify))
autoencode(mode,x::Array, sparsify=true) = decode(model,encode(model, gpu(x), sparsify)) |> data |> cpu
# model = Chain(
#                 Conv((7,1), 1 =>1k, relu, pad=(3,0)),
#                 Conv((7,1), 1k=>1k, relu, pad=(3,0)),
#                 Conv((7,1), 1k=>2k, relu, pad=(3,0)),
#                 Conv((7,1), 2k=>3k, relu, pad=(3,0)),
#                 Conv((7,1), 3k=>2k, relu, pad=(3,0)),
#                 Conv((7,1), 2k=>1k, relu, pad=(3,0)),
#                 Conv((7,1), 1k=>1k, relu, pad=(3,0)),
#                 Conv((7,1), 1k=>1,        pad=(3,0))
#                 )
function __init__()
    # The model definition must be done at init time since it contains pointers to CuArrays on the GPU, ref https://github.com/JuliaPy/PyCall.jl#using-pycall-from-julia-modules
    global model = Chain(
                    Conv((7,1), 1 =>1k, leakyrelu, pad=(2,0)),
                    Conv((11,1), 1k=>1k, leakyrelu, pad=(0,0)),
                    Conv((11,1), 1k=>1k, leakyrelu, pad=(0,0), stride=2),
                    Conv((11,1), 1k=>1k, leakyrelu, dilation=3, pad=(0,0), stride=3),
                    Conv((11,1), 1k=>4k,            pad=(0,0), stride=2),
                    ConvTranspose((11,1), 4k=>1k, leakyrelu, pad=(0,0), stride=3),
                    ConvTranspose((11,1), 1k=>1k, leakyrelu, dilation=3, pad=(0,0), stride=2),
                    ConvTranspose((11,1), 1k=>1k, leakyrelu, pad=(0,0), stride=2),
                    ConvTranspose((11,1),  1k=>1k, leakyrelu, pad=(0,0)),
                    ConvTranspose((24,1),  1k=>1,            pad=(0,0)),
                    ) |> gpu
end
