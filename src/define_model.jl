
using Flux: data
const k = 20
const k2 = 3

# generic ======================================================================
# function encode(model, X, sparsify)
#     cpu(data(encode(model, maybegpu(model,X), sparsify)))
# end
# function autoencode(model,x, sparsify)
#     decode(model,encode(model, maybegpu(model,x), sparsify)) |> data |> cpu
# end

const TrackedCuArray = TrackedArray{<:Any, <:Any, <:CuArray}
autoencode(model,x) = decode(model,encode(model, x))
ongpu(m) = m[1].bias.data isa CuArray
@inline maybegpu(model, x) = ongpu(model) ? gpu(x) : x
@inline maybegpu(model, x::CuArray) = x
@inline maybegpu(model, x::TrackedCuArray) = x
@inline function maybegpu(model, xy::Tuple)
    ongpu(model) || return xy
    x,y = xy
    X   = gpu(x)
    Y   = gpu(y)
    (X,Y)
end

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
preM(Z::TrackedCuArray) where T = gpu(preM_(Z))
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

@inline ongpu(m::MixtureAutoencoder) = m.e[1].bias.data isa CuArray

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
    # λi = controller(λi, Flux.data(l), Flux.data(ie), 1.001f0)
    # λl = controller(λl, ltarget, Flux.data(le), 1/1.001f0)
    λi = controller(λi, ltarget/10, Flux.data(ie), 0.01f0)
    λl = controller(λl, ltarget, Flux.data(le), -0.01f0)
    model.weights = (λi,λl)
    push!(losses, Flux.data.((l/var(Y),ie,le)))
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


# ==============================================================================
# AutoEncoder

struct AutoEncoder{TE,TD}
    e::TE
    d::TD
    sparsify::Bool
end

Flux.@treelike AutoEncoder
(m::AutoEncoder)(x) = autoencode(m,x)
ongpu(m::AutoEncoder) = m.e[1].bias.data isa CuArray

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
    Zc = cpu(Flux.data(Z))
    sparsify_wta!(Zc)
    Z = gpu(Zc) .* Z
end

function loss(model::AutoEncoder, xy::Tuple, losses)
    X,Y = maybegpu(model, xy)
    Z   = encode(model, X)
    Xh  = decode(model, Z)
    l   = sum(abs.(robust_error(Y,Xh))) * Float32(1 / length(X))
    push!(losses, Flux.data(l/var(Y)))
    l
end


function encode(model::AutoEncoder, X)
    X = maybegpu(model,X)
    X = reshape(X, size(X,1), 1, 1, :)
    Z = model.e(X)
    model.sparsify ? oneactive(Z) : Z
end

decode(model::AutoEncoder, Z) = model.d(Z)


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
    push!(losses, (Flux.data(rl/var(Y)), Flux.data(cl)))
    rl, 0.01cl
end
encode(model::ResidualEncoder, X) = encode(model.ae, X)
decode(model::ResidualEncoder, Z) = decode(model.ae, Z)
function classify(model::ResidualEncoder, x)
    X  = maybegpu(model, x)
    Z  = encode(model, X)
    Xh = decode(model, Z)
    E  = robust_error(X,Xh)
    model.fc(reshape(E,:,1,1,size(E,2)))
end
classify(model::ResidualEncoder, x::Tuple) = classify(model, x[1])
