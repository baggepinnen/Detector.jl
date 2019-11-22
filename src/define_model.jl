
using Flux: data
const k = 20
const k2 = 3

@inline maybegpu(model, x) = ongpu(model) ? gpu(x) : x

struct MixtureAutoencoder{TE,TD,TM,TS}
    e::TE
    d::TD
    m::TM
    state::TS
end
Flux.@treelike MixtureAutoencoder
(m::MixtureAutoencoder)(x) = autoencode(m,x,true)

preM(Z) = vec(mapcols(norm,reshape(Z,:,size(Z,3))))
preM(Z::TrackedArray{<:Any, <:Any, <:CuArray}) where T = gpu(vec(mapcols(norm,reshape(Z,:,size(Z,3)))))

function MixtureAutoencoder(k)

    e = Chain(      Conv((11,1), 1 =>1k, leakyrelu, pad=(15,0), dilation=(3,1)),
                    Conv((11,1), 1k=>1k, leakyrelu, pad=(15,0), dilation=(3,1)),
                    Conv((11,1), 1k=>1k, leakyrelu, pad=(15,0), dilation=(3,1)),
                    Conv((11,1), 1k=>5k,            pad=(15,0), dilation=(3,1)),
                    )
    d = Conv((11,1), 5k=>5k,             pad=(15,0), dilation=(3,1))
    m = Chain(preM, Dense(5k, 5k, relu), Dense(5k, 5k, tanh), Dense(5k, 5k), softmax) # Dont put relu here to avoid outputting hard 0
    MixtureAutoencoder(e,d,m,fill(Float32(1/5k), 5k))
end

@inline ongpu(m::MixtureAutoencoder) = m.e[1].bias.data isa CuArray

encoderlength(model::MixtureAutoencoder) = length(model.ae) - 1
function encode(model::MixtureAutoencoder, X, _=true)
    X = reshape(maybegpu(model,X), size(X,1), 1, 1, :)
    Z = model.e(X)
    M = model.m(Z)
    (Z, M)
end
decode(model::MixtureAutoencoder, (Z, M), _=true) = reshape(model.d(Z), size(Z,1), :) * M

scalarent(x::Real) =                  -log(x + Float32(1e-6))*x
CuArrays.@cufunc scalarent(x::Real) = -log(x + Float32(1e-6))*x
vectorent(x::AbstractArray) = sum(scalarent, x)

function loss(model::MixtureAutoencoder, x)
    X = maybegpu(model, x)
    Z,M = encode(model, X)
    Xh  = decode(model, (Z,M))
    l   = sum(abs2.(robust_error(X,Xh))) * Float32(1 / length(X))
    le, state = longterm_entropy(M, model.state)
    model.state .= state
    ie = vectorent(M)
    # @show M
    # @show ((l, ie, le))
    l + ie #- le
end

function longterm_entropy(Zn, state, λ=0.999f0)
    state = max.(Zn, λ*state) |> x-> x./sum(x)
    # ent   = sum(state .* log.(state))
    ent   = vectorent(state)
    ent, Flux.data(state)
end

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
encoderlength(model::MixtureAutoencoder) = length(model)-1

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

function oneactive(Z)
    Zc = cpu(Flux.data(Z))
    sparsify_wta!(Zc)
    Z = gpu(Zc) .* Z
end

# generic ======================================================================
# function encode(model, X, sparsify)
#     cpu(data(encode(model, maybegpu(model,X), sparsify)))
# end
# function autoencode(model,x, sparsify)
#     decode(model,encode(model, maybegpu(model,x), sparsify)) |> data |> cpu
# end

autoencode(model,x, sparsify) = decode(model,encode(model, x, sparsify))
# ==============================================================================
# Standard model
ongpu(m) = m[1].bias.data isa CuArray

function loss(model, x)
    X  = gpu(x)
    Z  = encode(model, X, sparsify)
    Zn = slicemap(norm, Z, dims=(1,2)) |> softmax
    Z = Z .* Zn
    Xh = decode(model, Z)
    l = sum(abs2.(robust_error(X,Xh))) * 1 // length(X)
    if sparsify && rand() < 0.5
        # l2 = zero(l)
        # for b in 1:size(Z,4), c in 1:size(Z,3)
        #     l2 += sum(gpu([λi]).*norm(@view(Z[:,:,c,b])))
        # end
        return l - entropy(Zn) - longterm_entropy(Zn)

        # l += sum(gpu([λi]).*sum(abs,Z)/ size(Z,3))
        # l2 = sum(norm(@view(Z[:,:,c])) c in 1:size(Z,3)*size(Z,4))
        # l += l2 * (1 // size(Z,3))
    end
    l
end


encoderlength(model) = length(model) ÷ 2
function encode(model, X, sparsify)
    X = reshape(X, size(X,1), 1, 1, :)
    Z = model[1:encoderlength(model)](X)
    # sparsify ? oneactive(Z) : Z
    Z
end

decode(model, Z) = model[encoderlength(model)+1:end](Z)
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

     # @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae"

    # The model definition must be done at init time since it contains pointers to CuArrays on the GPU, ref https://github.com/JuliaPy/PyCall.jl#using-pycall-from-julia-modules
    # global model = Chain(
    #                 Conv((7,1), 1 =>1k, leakyrelu, pad=(2,0)),
    #                 Conv((11,1), 1k=>1k, leakyrelu, pad=(0,0)),
    #                 Conv((11,1), 1k=>1k, leakyrelu, pad=(0,0), stride=2),
    #                 Conv((11,1), 1k=>1k, leakyrelu, dilation=3, pad=(0,0), stride=3),
    #                 Conv((11,1), 1k=>4k,            pad=(0,0), stride=2),
    #                 ConvTranspose((11,1), 4k=>1k, leakyrelu, pad=(0,0), stride=3),
    #                 ConvTranspose((11,1), 1k=>1k, leakyrelu, dilation=3, pad=(0,0), stride=2),
    #                 ConvTranspose((11,1), 1k=>1k, leakyrelu, pad=(0,0), stride=2),
    #                 ConvTranspose((11,1),  1k=>1k, leakyrelu, pad=(0,0)),
    #                 ConvTranspose((24,1),  1k=>1,            pad=(0,0)),
    #                 )


    global model = Chain(
                    Conv((11,1), 1 =>1k2,  leakyrelu, pad=(15,0), dilation=(3,1)),
                    Conv((11,1), 1k2=>1k2, leakyrelu, pad=(15,0), dilation=(3,1)),
                    Conv((11,1), 1k2=>1k2, leakyrelu, pad=(15,0), dilation=(3,1)),
                    Conv((11,1), 1k2=>5k2,            pad=(15,0), dilation=(3,1)),
                    Conv((11,1), 5k2=>1,      pad=(15,0), dilation=(3,1)),
                    )
end
