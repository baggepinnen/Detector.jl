
using Flux: data
const k = 5

function oneactive(Z)
    Zc = cpu(Flux.data(Z))
    a = argmax(abs.(Zc), dims=1)
    Zc .= 0
    Zc[a] .= 1
    Z = Flux.data(gpu(Zc)) .* Z
end
encode(model, X::Array, sparsify=true) = cpu(data(encode(model, gpu(X), sparsify)))
encoderlength(model) = length(model) ÷ 2
# encoderlength(model) = length(model)-2
function encode(model, X, sparsify=true)
    X = reshape(X, :, 1, 1, size(X,4))
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
