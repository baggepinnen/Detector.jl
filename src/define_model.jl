

# generic ======================================================================
# function encode(model, X, sparsify)
#     cpu(identity(encode(model, maybegpu(model,X), sparsify)))
# end
# function autoencode(model,x, sparsify)
#     decode(model,encode(model, maybegpu(model,x), sparsify)) |> identity |> cpu
# end

# const TrackedCuArray = TrackedArray{<:Any, <:Any, <:CuArray}

"""
    autoencode(model, x)

Reconstruct `x` using `model`
"""
autoencode(model,x) = decode(model,encode(model, x))

"""
    classify(model::ResidualEncoder, x::Tuple)

Use the classification part of the encoder to classify an input
"""
classify(model, x::Tuple) = classify(model, x[1])
function classify(identityset)
    c = map(identityset) do (x,y,l)
        classify(model,x)' |> cpu
    end
    vec(reduce(vcat, c))
end
# ongpu(m) = m[1].bias.identity isa CuArray

"""
    ongpu(model)

Determine whether a model resides on the gpu or not.
"""
ongpu(m) = m[1].bias isa CuArray

"""
    maybegpu(model, x)

Put `x` on the gpu if the model is on the gpu
"""
@inline maybegpu(model, x) = ongpu(model) ? gpu(x) : x
@inline maybegpu(model, x::CuArray) = x
# @inline maybegpu(model, x::TrackedCuArray) = x
@inline function maybegpu(model, xy::Tuple)
    ongpu(model) || return xy
    x,y = xy
    X   = gpu(x)
    Y   = gpu(y)
    (X,Y)
end

@inline maybegpu(model, xy::Tuple{Vararg{<:CuArray}}) = xy

"""
    loss(model, x, losses)

Loss function specific to each kind of `model`.

#Arguments:
- `x`: Input
- `losses`: A vector to store the loss value in, the eltype depends on the model.
"""
function loss(model, x, losses)
    X = maybegpu(model, x)
    loss(model, (X,X), losses)
end

# Mixture ======================================================================
mutable struct MixtureAutoencoder{TE,TD,TM,TS,TW}
    e::TE
    d::TD
    m::TM
    state::TS
    weights::TW
end
Flux.@treelike MixtureAutoencoder
(m::MixtureAutoencoder)(x) = autoencode(m,x)

preM_(Z) = reshape(mapcols(norm,reshape(Z,size(Z,1),:)), size(Z,3), size(Z,4))
preM(Z) = preM_(Z)
# preM(Z::TrackedCuArray) where T = gpu(preM_(Z))
preM(Z::CuArray) where T = gpu(preM_(Z))
# Base.delete_method.(methods(preM))
function MixtureAutoencoder(k)

    e = Chain(  Conv((7,1), 1 =>1k, leakyrelu, pad=(0,0)),
                BatchNorm(1k),
                Conv((11,1), 1k=>1k, leakyrelu, pad=(0,0)),
                BatchNorm(1k),
                Conv((11,1), 1k=>1k, leakyrelu, pad=(0,0), stride=2),
                BatchNorm(1k),
                Conv((11,1), 1k=>1k, leakyrelu, dilation=4, pad=(0,0), stride=2),
                BatchNorm(1k),
                Conv((11,1), 1k=>5k,            pad=(0,0), stride=2),
                    )
    d = Chain(ConvTranspose((11,1), 5k=>5k, leakyrelu, stride=2),
                    ConvTranspose((13,1), 5k=>5k, leakyrelu, dilation=3, stride=2),
                    BatchNorm(5k),
                    ConvTranspose((13,1), 5k=>5k, leakyrelu, stride=2),
                    BatchNorm(5k),
                    ConvTranspose((13,1),  5k=>5k, leakyrelu),
                    BatchNorm(5k),
                    ConvTranspose((16,1),  5k=>5k))
    m = Chain(preM, Dense(5k, 5k, relu), Dense(5k, 5k, tanh), Dense(5k, 5k), softmax) # Dont put relu here to avoid outputting hard 0
    MixtureAutoencoder(e,d,m,Flux.param(fill(Float32(1/5k), 5k)), (40.0f0,40.0f0))
end

@inline ongpu(m::MixtureAutoencoder) = m.e[1].bias isa CuArray

function encode(model::MixtureAutoencoder, X)
    X = reshape(maybegpu(model,X), size(X,1), 1, 1, :)
    Z = model.e(X)
    M = model.m(Z)
    (Z, M)
end

function decode(model::MixtureAutoencoder, (Z, M))
    Zd = dropdims(model.d(Z), dims=2)
    # @mul E[a,d] := Zd[a,b,d]*M[b,d]
    Xh = dropdims(sum(Zd.*reshape(M,1,size(M)...), dims=2), dims=2)
    Xh = reshape(Xh, size(Xh,1), 1, 1, :)
end

scalarent(x::Real) =                  -log(x + Float32(1e-6))*x
CuArrays.@cufunc scalarent(x::Real) = -log(x + Float32(1e-6))*x
vectorent(x::AbstractArray) = sum(scalarent, x)

function loss(model::MixtureAutoencoder, xy::Tuple, losses)
    X,Y         = maybegpu(model, xy)
    Z,M         = encode(model, X)
    Xh          = decode(model, (Z,M))
    # @show size.((X,x,Z,M,Xh))
    l           = sum(abs2.(robust_error(Y,Xh))) * Float32(1 / length(X))
    le, state   = longterm_entropy(M, model.state)
    model.state = state
    ie          = mean(vectorent(M) for M in eachcol(M))
    λi,λl       = model.weights
    ltarget = Float32(scalarent(1/size(M,1))*size(M,1)/2)
    # λi = controller(λi, identity(l), identity(ie), 1.001f0)
    # λl = controller(λl, ltarget, identity(le), 1/1.001f0)
    λi = controller(λi, ltarget/10, identity(ie), 0.01f0)
    λl = controller(λl, ltarget, identity(le), -0.01f0)
    model.weights = (λi,λl)
    pushlog!(losses, (l/var(Y),ie,le))
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
        error("This does not work for Zygote")
        state = state
        counter = 0
    end
    # ent   = sum(state .* log.(state))
    ent, state
end


# ==============================================================================
# AutoEncoder

struct AutoEncoder{TE,TD}
    e::TE
    d::TD
    sparsify::Bool
end

Flux.@treelike AutoEncoder
(m::AutoEncoder)(x) = autoencode(m,x)
ongpu(m::AutoEncoder) = m.e[1].bias isa CuArray

function sparsify_wta!(Zc)
    @inbounds for bi in 1:size(Zc,4)
        for ci in 1:size(Zc,3)
            a = argmax(abs.(@view(Zc[:,:,ci,bi]))[:])
            Zc[:,:,ci,bi] .= Float32(0.01 .* randn.())
            Zc[a,:,ci,bi] .= 1
        end
    end
end


# function sparsify_channels!(Zc)
#     @inbounds for bi in 1:size(Zc,4)
#         i = 0
#         am = 0.
#         for ci in 1:size(Zc,3)
#             a = sum(abs, @view(Zc[:,:,ci,bi]))
#             if a > am
#                 am = a
#                 i = ci
#             end
#         end
#         rand() < 0.01 && @show i
#         # acts = vec(sum(abs, @view(Zc[:,:,:,bi]), dims=1))
#         # i = argmax(acts)
#         Zc[:,:,:,bi] .= 0.1randn(size(Zc)[1:3]...)
#         Zc[:,:,i,bi] .= 1
#     end
# end

function oneactive(Z)
    Zc = cpu(Z)
    sparsify_wta!(Zc)
    Z = gpu(Zc) .* Z
end

function loss(model::AutoEncoder, xy::Tuple, losses)
    X,Y = maybegpu(model, xy)
    Z   = encode(model, X)
    Xh  = decode(model, Z)
    l   = sum(abs.(robust_error(Y,Xh))) * Float32(1 / length(X))
    pushlog!(losses, l/var(Y))
    l
end


"""
    Z = encode(model, X)

Encode `X` into the latent space `Z`
"""
function encode(model::AutoEncoder, X)
    X = maybegpu(model,X)
    X = reshape(X, size(X,1), 1, 1, :)
    Z = model.e(X)
    model.sparsify ? oneactive(Z) : Z
end

"""
    Xh = decode(model, Z)

Decode latent space representation `Z` to form reconstruction `Xh`
"""
decode(model::AutoEncoder, Z) = model.d(Z)


"""
    AutoEncoder(k; sparsify=true)

`k` controls the size of the model, 1 is small, 20+ is large.
"""
function AutoEncoder(k; sparsify=true)

     # @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae"

    # The model definition must be done at init time since it contains pointers to CuArrays on the GPU, ref https://github.com/JuliaPy/PyCall.jl#using-pycall-from-julia-modules
    e = Chain(
            Conv((51,1), 1 =>1k, leakyrelu, pad=(1,0)),
            Conv((21,1), 1k=>1k, leakyrelu, pad=(1,0)),
            Conv((11,1), 1k=>1k, leakyrelu, stride=3),
            BatchNorm(1k),
            Conv((11,1), 1k=>1k, leakyrelu, dilation=3, pad=(0,0), stride=3),
            Conv((11,1), 1k=>4k,            pad=(0,0), stride=3))
    d = Chain(
            ConvTranspose((11,1), 4k=>1k, leakyrelu, pad=(0,0), stride=3),
            ConvTranspose((11,1), 1k=>1k, leakyrelu, dilation=3, pad=(0,0), stride=3),
            BatchNorm(1k),
            ConvTranspose((11,1), 1k=>1k, leakyrelu, pad=(0,0), stride=3),
            ConvTranspose((21,1),  1k=>1k, leakyrelu, pad=(0,0)),
            ConvTranspose((48,1),  1k=>1,            pad=(0,0)))


    # global model = Chain(
    #                 Conv((11,1), 1 =>1k2,  leakyrelu, pad=(15,0), dilation=(3,1)),
    #                 Conv((11,1), 1k2=>1k2, leakyrelu, pad=(15,0), dilation=(3,1)),
    #                 Conv((11,1), 1k2=>1k2, leakyrelu, pad=(15,0), dilation=(3,1)),
    #                 Conv((11,1), 1k2=>5k2,            pad=(15,0), dilation=(3,1)),
    #                 Conv((11,1), 5k2=>1,      pad=(15,0), dilation=(3,1)),
    #                 )
    AutoEncoder(e,d,sparsify)
end



# Residual encoder =============================================================
batchvec(x) = reshape(x,:,size(x,ndims(x)))

struct ResidualEncoder
    ae
    fc
end

(m::ResidualEncoder)(x) = autoencode(m,x)

"""
    ResidualEncoder(k::Int, k2::Int; sparsify=true)

Creates a `ResidualEncoder`. This model has a classifier as well as an autoencoder. `k` controls the size and `k2` must be chosen so that sizes of internal vectors lign up, you can determine this by trying to classify an input and observing the error message.

#Arguments:
- `k`: Size of model
- `k2`: size of internal representation
- `sparsify`: whether or not to sparsify the latent space of the autoencoder.
"""
function ResidualEncoder(k::Int, k2::Int; sparsify=true)
    ae = AutoEncoder(k; sparsify=sparsify)
    fc = Chain(
                Conv((21,1), 1 =>1k, leakyrelu),
                MaxPool((6,1)),
                BatchNorm(1k),
                Conv((11,1), 1k =>1k, leakyrelu, stride=4),
                MaxPool((6,1)),
                BatchNorm(1k),
                Conv((11,1), 1k=>1k, leakyrelu, stride=4),
                MaxPool((6,1)),
                BatchNorm(1k),
                Conv((11,1), 1k=>1k, leakyrelu, stride=4),
                batchvec,
                Dense(k2,10k,relu),
                BatchNorm(10k),
                Dense(10k,5k,tanh),
                Dense(5k,1,σ),
    )
    ResidualEncoder(ae,fc)
end

Flux.@treelike ResidualEncoder
ongpu(m::ResidualEncoder) = ongpu(m.ae)
CuArrays.@cufunc Flux.binarycrossentropy(ŷ, y; ϵ=eps(ŷ)) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)
function loss(model::ResidualEncoder, xyl::Tuple, losses)
    X,Y = maybegpu(model, (xyl[1], xyl[2]))
    lab = xyl[3]
    Z   = encode(model, X)
    Xh  = decode(model, Z)
    E   = robust_error(Y,Xh)
    rl  = sum(abs.(E)) * Float32(1 / length(X))
    cl  = sum(Flux.binarycrossentropy.(model.fc(reshape(E,:,1,1,size(E,2))), lab))
    pushlog!(losses, (rl/var(Y), cl))
    rl, 0.01cl
end
encode(model::ResidualEncoder, X) = encode(model.ae, X)
decode(model::ResidualEncoder, Z) = decode(model.ae, Z)
function classify(model::ResidualEncoder, x::AbstractArray)
    X  = maybegpu(model, x)
    Z  = encode(model, X)
    Xh = decode(model, Z)
    E  = robust_error(X,Xh)
    model.fc(reshape(E,:,1,1,size(E,2)))
end







# Variational autoencoder ======================================================

mutable struct VAE{TE,TD}
    e::TE
    d::TD
    c::Float32
    ci::Float32
end
Flux.@treelike VAE
ongpu(m::VAE) = ongpu(m.e)

(m::VAE)(x) = autoencode(m,x)

"""
    VAE(k::Int; c0=0.01, ci=0.001)

Create a Variational Autoencoder.

#Arguments:
- `k`: Size of the model, 1 is small, 15+ is large.
- `c0`: Initial KL penalty
- `ci`: Increment to KL penalty per iteration. Caps at 1.
"""
function VAE(k::Int; c0=.01, ci=0.001)
    e = Chain(
            Conv((51,1), 1 =>4k, leakyrelu, pad=(0,0)),
            BatchNorm(4k),
            Conv((21,1), 4k=>3k, leakyrelu, pad=(0,0)),
            BatchNorm(3k),
            Conv((11,1), 3k=>2k, leakyrelu, stride=2),
            BatchNorm(2k),
            Conv((11,1), 2k=>1k, leakyrelu, pad=(0,0), stride=2),
            BatchNorm(1k),
            Conv((11,1), 1k=>2,            pad=(0,0), stride=2))
    d = Chain(
            ConvTranspose((11,1), 1=>1k, leakyrelu, pad=(0,0), stride=2),
            BatchNorm(1k),
            ConvTranspose((11,1), 1k=>2k, leakyrelu, pad=(0,0), stride=2),
            BatchNorm(2k),
            ConvTranspose((11,1), 2k=>3k, leakyrelu, pad=(0,0), stride=2),
            BatchNorm(3k),
            ConvTranspose((21,1),  3k=>4k, tanh, pad=(0,0)),
            BatchNorm(4k),
            ConvTranspose((54,1),  4k=>2,            pad=(0,0)))
    m = VAE(e,d,Float32(c0),Float32(ci))
    m.d[end].bias[2] = 1
    m.e[end].bias[2] = -2
    m
end

function loss(model::VAE, xy::Tuple, losses)
    X,Y  = maybegpu(model, xy)
    Z    = encode(model, X)

    Xh,Σ = decode(model, Z)
    E    = robust_error(Y,Xh)
    # rl   = sum(abs2.(E)) / length(X)
    rl   = varloss(E,Σ)
    kll  = kl(Z)
    model.c = min(1, model.c + model.ci)
    pushlog!(losses, (sum(abs2.(E)) / length(X) /var(Y), kll))
    rl, model.c*kll
end
encode(model::VAE, X) = model.e(X)

"""
    decode(model::VAE, Z, noise=true)

#Arguments:
- `noise`: Whether or not to sample from the latent distribution or onlr propagate the mean.
"""
function decode(model::VAE, Z, noise=true)
    if noise
        Z = decode_kernel.(Z[:,:,1:1,:], Z[:,:,2:2,:], CuArrays.randn(Float32, size(Z,1), size(Z,2), 1, size(Z,4)))
        # Z = decode_kernel.(Z[:,:,1:1,:], Z[:,:,2:2,:])
    else
        Z = Z[:,:,1:1,:]
    end
    XΣ = model.d(Z)
    XΣ[:,:,1:1,:], XΣ[:,:,2:2,:]
end

"""
    autoencode(model::VAE, x, noise=true) = begin

DOCSTRING

#Arguments:
- `noise`: Whether or not to sample from the latent distribution or onlr propagate the mean.
"""
autoencode(model::VAE, x, noise=true) = decode(model, encode(model,x), noise)

"""
    kl(Z)

Claculate the KL divergence penalty for latent space representation `Z` of a `VAE`
"""
function kl(Z)
    μ = Z[:,:,1,:]
    lσ = Z[:,:,2,:]
    # 0.5f0*(sum(abs2,μ) + sum(exp,lσ)) - sum(lσ)
    (0.5f0*sum(abs2.(μ) .+ exp.(lσ)) - sum(lσ))/ length(μ)
end

function varloss(E,Σ)
    (0.5f0*sum(abs2.(E)./(exp.(Σ) .+ 1f-2)) + sum(Σ))/length(E)
end

decode_kernel(Z1,Z2,r) = Z1 + sqrt(exp(Z2))*r
CuArrays.@cufunc decode_kernel(Z1,Z2,r) = Z1 + exp(Z2/2)*r
CuArrays.@cufunc decode_kernel(Z1,Z2) = Z1 + exp(Z2/2)*randn()
#
# kl_kernel(μ, σ, lσ) = -lσ + (σ + abs2(μ))/(σ2 + 1f-5)
# CuArrays.@cufunc kl_kernel(μ, σ, lσ) = -lσ + (σ + μ^2)/(σ2 + 1f-5)
#
# function gpukl(l,μ, lσ)
#     for i in 1:length(μ)
#         @inbounds l[1] += -lσ[i] + (exp(lσ[i]) + μ[i]^2)/(exp(lσ[i]) + 1f-5)
#     end
#     nothing
# end

# μ1, σ1² = stats(e)
# lσ1 = log.(sqrt.(σ1²))
# lσ2 = log.(σ2)#log.(sqrt.(var(dy)))
# l = 0f0
# for i = eachindex(μ1)
#     l += c*2lσ2[i] - 2lσ1[i] +
#     c*(σ1²[i] + abs2(μ1[i] - μ2[i]))/(σ2[i]^2 + 1f-5)
# end
# 0.5f0l
