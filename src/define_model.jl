
using Flux: data
const k = 20
const k2 = 3

@inline maybegpu(model, x) = ongpu(model) ? gpu(x) : x

Base.@kwdef mutable struct MixtureAutoencoder{TE,TD,TM,TS,TW}
    e::TE
    d::TD
    m::TM
    state::TS
    weights::TW = (10.0f0,40.0f0)
end
Flux.@treelike MixtureAutoencoder
(m::MixtureAutoencoder)(x) = autoencode(m,x,true)

preM_(Z) = reshape(mapcols(norm,reshape(Z,size(Z,1),:)), size(Z,3), size(Z,4))
preM(Z) = preM_(Z)
preM(Z::TrackedArray{<:Any, <:Any, <:CuArray}) where T = gpu(preM_(Z))
# Base.delete_method.(methods(preM))
function MixtureAutoencoder(k)

    e = Chain(  Conv((7,1), 1 =>1k, leakyrelu, pad=(0,0)),
                Conv((11,1), 1k=>1k, leakyrelu, pad=(0,0)),
                Conv((11,1), 1k=>1k, leakyrelu, pad=(0,0), stride=2),
                Conv((11,1), 1k=>1k, leakyrelu, dilation=4, pad=(0,0), stride=2),
                Conv((11,1), 1k=>5k,            pad=(0,0), stride=2),
                    )
    d = Chain(ConvTranspose((11,1), 5k=>5k, leakyrelu, stride=2),
                    ConvTranspose((13,1), 5k=>5k, leakyrelu, dilation=3, stride=2),
                    ConvTranspose((13,1), 5k=>5k, leakyrelu, stride=2),
                    ConvTranspose((13,1),  5k=>5k, leakyrelu),
                    ConvTranspose((16,1),  5k=>5k))
    m = Chain(preM, Dense(5k, 5k, relu), Dense(5k, 5k, tanh), Dense(5k, 5k), softmax) # Dont put relu here to avoid outputting hard 0
    MixtureAutoencoder(e,d,m,Flux.param(fill(Float32(1/5k), 5k)), (10.0f0,40.0f0))
end

@inline ongpu(m::MixtureAutoencoder) = m.e[1].bias.data isa CuArray

encoderlength(model::MixtureAutoencoder) = length(model.ae) - 1
function encode(model::MixtureAutoencoder, X, _=true)
    X = reshape(maybegpu(model,X), size(X,1), 1, 1, :)
    Z = model.e(X)
    M = model.m(Z)
    (Z, M)
end
using TensorCast
function decode(model::MixtureAutoencoder, (Z, M), _=true)
    Zd = dropdims(model.d(Z), dims=2)
    # @mul E[a,d] := Zd[a,b,d]*M[b,d]
    dropdims(sum(Zd.*reshape(M,1,size(M)...), dims=2), dims=2)
end

scalarent(x::Real) =                  -log(x + Float32(1e-6))*x
CuArrays.@cufunc scalarent(x::Real) = -log(x + Float32(1e-6))*x
vectorent(x::AbstractArray) = sum(scalarent, x)

function loss(model::MixtureAutoencoder, x, losses)
    X           = maybegpu(model, x)
    Z,M         = encode(model, X)
    Xh          = decode(model, (Z,M))
    l           = sum(abs2.(robust_error(X,Xh))) * Float32(1 / length(X)/size(X,4))
    le, state   = longterm_entropy(M, model.state)
    model.state = state
    ie          = mean(vectorent(M) for M in eachcol(M))
    λi,λl       = model.weights
    ltarget = Float32(scalarent(1/size(M,1))*size(M,1)/2)
    # λi = controller(λi, Flux.data(l), Flux.data(ie), 1.001f0)
    # λl = controller(λl, ltarget, Flux.data(le), 1/1.001f0)
    λi = controller(λi, ltarget/10, Flux.data(ie), 0.001f0)
    λl = controller(λl, ltarget, Flux.data(le), -0.001f0)
    # @show model.weights = (λi,λl)
    push!(losses, Flux.data.((l/var(x),ie,le)))
    l, λi*ie, -λl*le
    # l, ie, -4*le
    # l, ie, -le
end

# controller(λ,a,b,k) = max(b > a ? k*λ : λ/k, 0.001f0)
controller(λ,a,b,k) = clamp(λ - k*(a-b), 0.001f0, 100f0)

counter = 0
function longterm_entropy(Zn, state, λ=0.95f0)
    global counter += 1
    # for Zn in eachcol(Zn)
    #     state = (1-λ)*Zn + λ*state |> x-> x./sum(x)
    # end
    state = (1-λ)*vec(mean(Zn, dims=2)) + λ*state |> x-> x./sum(x)
    ent   = vectorent(state)
    # state.tracker.f = Tracker.Call(nothing, ())
    if counter >= 50
        state = Flux.param(state.data)
        counter = 0
    end
    # ent   = sum(state .* log.(state))
    ent, state
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

function loss(model, x, losses)
    X  = gpu(x)
    Z  = encode(model, X, true)
    Xh = decode(model, Z)
    l = sum(abs2.(robust_error(X,Xh))) * 1 // length(X)
    push!(losses, Flux.data(l/var(x)))
    l
end


encoderlength(model) = length(model) ÷ 2
function encode(model, X, sparsify)
    X = maybegpu(model,X)
    X = reshape(X, size(X,1), 1, 1, :)
    Z = model[1:encoderlength(model)](X)
    sparsify ? oneactive(Z) : Z
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
    #                 Conv((11,1), 1k=>1k, leakyrelu, stride=2),
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
