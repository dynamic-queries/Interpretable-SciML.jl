using DifferentialEquations
using DataInterpolations
using Plots
using ForwardDiff

#===============================================================#
abstract type DynamicalSystem end

struct LorenzSystem <: DynamicalSystem end
struct LotkaVolterra <: DynamicalSystem end
struct NSECylinder <: DynamicalSystem end
struct HelicopterData <: DynamicalSystem end

function datagen(data::LorenzSystem)
    function noise(u)
        for i = 1:length(u)
            for j = 1:length(u[i])
                u[i][j] += 0.1*randn()
            end
        end
        return u
    end

    function lorenz!(du,u,p,t)
        du[1] = p[1]*(u[2]-u[1])
        du[2] = u[1]*(p[2]-u[3]) - u[2]
        du[3] = u[1]*u[2] - p[3]*u[3]
    end

    u₀ = [1,0,0]
    p = [10,12,8/3] # Change d_lorenz
    tspan = (0,100)
    t = 0.01:0.01:100
    prob = ODEProblem(lorenz!,u₀,tspan,p)
    sol = solve(prob,Tsit5(),saveat=t)
    # noisy = noise(sol.u)
    fig = plot(sol,vars=(1,2,3))
    display(fig)
    return sol
end

#===============================================================#

function denoise!(sol)
    x_shift = circshift(sol, (0,-1))
    y_shift = circshift(sol, (-1,0))
    x_diff = sol - x_shift
    y_diff = sol - y_shift
    grad_norm2 = x_diff ^ 2 + y_diff ^ 2 + 1.1920929e-07
    norm = sum(sqrt.(grad_norm2))
    dgrad_norm = 0.5 / sqrt.(grad_norm2)
    dx_diff = 2 * x_diff * dgrad_norm
    dy_diff = 2 * y_diff * dgrad_norm
    grad = dx_diff + dy_diff
    grad[:, 1:end, :] -= dx_diff[:, 1:-1, :]
    grad[1:end, :, :] -= dy_diff[1:-1, :, :]

    return norm, grad

end

#===============================================================#

function munge(A::Vector{Vector{Float64}})
    l1 = length(A)
    l2 = length(A[1])
    M = zeros(l2,l1)
    for  i = 1:l1
        M[:,i] = A[i]
    end
    M
end

#===============================================================#

abstract type AbstractDiff  end

struct AnalyticalDeriv <: AbstractDiff end
struct FiniteDiff <: AbstractDiff end
struct TotalVariationalDerivativative <: AbstractDiff end

function differentiate(sol,o::TotalVariationalDerivativative)
    # interp = LagrangeInterpolation(munge(sol.u),sol.t)
    interp = QuadraticInterpolation(munge(sol.u),sol.t)
    y_reconstruct = interp.(sol.t)
    v = [DataInterpolations.derivative(interp,k) for k in sol.t]
    temp = munge(v)
    display(plot(temp[1,:],temp[2,:],temp[3,:]))
    return v
end


function differentiate(sol,o::FiniteDiff)
    q = sol.u[2:end] .- sol.u[1:end-1]
    q = q./(sol.t[2] - sol.t[1])
    w = munge(q)
    fig = plot(w[1,:],w[2,:],w[3,:])
    display(fig)
    return w
end

function differentiate(sol,ds::LorenzSystem,o::AnalyticalDeriv)
    function lorenz(u) # Modify when changing parameters
        u1 = 10*(u[2] - u[1])
        u2 = u[1]*(12 - u[3]) - u[2]
        u3 = u[1]*u[2] - 8/3 * u[3]
        return [u1 u2 u3]
    end
    deriv = []
    for i = 1:length(sol.u)
        append!(deriv,lorenz(sol.u[i]))
    end
    deriv = reshape(deriv,(3,length(sol.u)))
    fig = plot(deriv[1,:],deriv[2,:],deriv[3,:])
    display(fig)
    return deriv
end

#===============================================================#
function n2_terms(p::Int)
    return Int(p*(p+1)/2)
end

function quadratic(X::Array)
    """
        Assume data is passed in the form x,y,z
        Solve this issue for an array of arbitrary size.
    """
    s = size(X) # 3,1001
    p = s[1]
    s = s[2]
    n = n2_terms(p) # 6
    C = []
    for i=1:p
        for j=i:p
            append!(C,X[i,:].*X[j,:])
        end
    end
    reshape(C,(n,s))
end

function cubic(X::Array)
    """
        Assume data is passed in the form x,y,z
        Solve this issue for an array of arbitrary size.
    """
    s = size(X) # 3,1001
    p = s[1]
    s = s[2]
    n = n2_terms(p) # 6
    C = []
    for i=1:p
        for j=i:p
            for k = j:p
                append!(C,X[i,:].*X[j,:] .* X[k,:])
            end
        end
    end
    # return C
    reshape(C,(10,s))
end




function linear(X::Array)
    """
        Assuming that the data passed is linear.
    """
    return X
end

function basis(X)
    s = size(X)
    ntsteps = s[2]
    nparams = s[1]
    # Since we consider unit, linear and quadratic terms.
    n = 1 + 3 + n2_terms(nparams) + 10 # You are better than this!
    θ = zeros(n,ntsteps)
    θ[1,:] = ones(ntsteps)
    θ[2:4,:] = linear(X)
    θ[5:10,:] = quadratic(X)
    θ[11:end,:] = cubic(X)
    return θ
end


# X = rand(3,1000)
# cubic(X)
# basis(X)
#===============================================================#
abstract type AbstractOptimizer end

struct LSTSQ <: AbstractOptimizer
    abstol::Float64
    LSTSQ() = new(1e-16)
end

struct STLSQ <: AbstractOptimizer
    abstol::Float64
    λ::Float64
    STLSQ(thres::Float64) = new(1e-16,thres)
end

##TODO : SR3

function _optimize(θ::Matrix{T},v::Vector{Vector{T}},opt::LSTSQ) where T
    θ = permutedims(θ,(2,1))
    v = permutedims(munge(v))
    return θ\v
end

function _optimize(θ::Matrix{T},v::Matrix{T},opt::LSTSQ) where T
    θ = permutedims(θ,(2,1))
    v = permutedims(v)
    return θ\v
end

function _optimize(θ::Matrix{T},v::Matrix{T},opt::STLSQ) where T
    iter = 0
    abstol = 1e-8
    θ = permutedims(θ,(2,1))
    v = permutedims(v)
    maxiter = maximum(collect(size(θ)))
    convlimit = abstol # Is the limit on the difference betweem two iterations.
    conv = Inf

    λ = opt.λ
    ξ = θ\v
    smallnums = abs.(ξ) .< λ
    bignums = @. !smallnums[:,1]
    x = similar(ξ)

    for i = 1:maxiter
        smallnums .= abs.(ξ) .< λ
        bignums .= @. !smallnums[:,1]
        y = similar(ξ) # Iteration level least square estimate.
        ξ[smallnums] .= 0
        # θ[:,bignums] * ξ[bignums,i] = v[:,i]
        for i = 1:size(v,2)
            bignums .= @. !smallnums[:,i]
            ξ[bignums,i] .= θ[:,bignums] \ v[:,i]
        end
        conv = LinearAlgebra.norm2(y - ξ)
        display(conv)
    end
    ξ[smallnums] .= 0
    return ξ
end

function _optimize(θ::Matrix{T},v::Vector{Vector{T}},opt::STLSQ) where T
    _optimize(θ,munge(v),opt)
end


#===============================================================#

struct Basis
    unit(x) = 1
    linear(x) = x
    square(x) = x^2
    q_mixed(x,y) = x*y
    cubic(x) = x^3
    c_mix1(x,y) = x
end

function _remake(ξ,ds::LorenzSystem)

    function make(du,u,p,t)

    end
end


#===============================================================#

# Main function

# Trail 1 : TVD Derivative functions
function trail1()
    obj_lorenz = LorenzSystem()
    sol = datagen(obj_lorenz)
    denoise!(sol)
    diff = TotalVariationalDerivativative()
    v = differentiate(sol,diff)
    X = munge(sol.u)
    θ = basis(X)
    opt = LSTSQ()
    ξ = _optimize(θ,v,opt)
end


# Trail 2 : Finite Diff Derivative
function trail2()
    obj_lorenz = LorenzSystem()
    sol = datagen(obj_lorenz)
    denoise!(sol)
    diff = FiniteDiff()
    w = differentiate(sol,diff)
    X = munge(sol.u)[:,2:1001]
    θ = basis(X)
    opt = LSTSQ()
    ξ = _optimize(θ,w,opt)
end


# Trail 3 : AnalyticalDeriv and the STLSQ optimizer.
function trail3()
    obj_lorenz = LorenzSystem()
    sol = datagen(obj_lorenz)
    denoise!(sol)
    # diff = AnalyticalDeriv()
    # d = Matrix{Float64}(differentiate(sol,obj_lorenz,diff))
    diff = TotalVariationalDerivativative()
    d = differentiate(sol,diff)
    X = munge(sol.u)
    θ = basis(X)
    display(LinearAlgebra.cond(θ))
    opt = STLSQ(0.05)
    ξ = _optimize(θ,d,opt)
    display(ξ)
    return θ,ξ
end

# Conclusion : Optimizer works!!
θ,ξ = trail3()

ξ











LinearAlgebra.norm2(θ)

#TODO: Reimplement the construction of the library function
#   - Test based implmentation of the library function with unit tests along every step of the way.

#TODO:  Automate the remake function
#   - List of basis  functions
#   - Every velocity vector is a linear combination of all these functions taken together.
#   - Use lambda functions
#   - This needs to be brute forced for the moment. Think of a better approach

#TODO: Implement the sparse regressor SR3
#   - This is a wish.
#   - The LSTSQ might be failing, but it is hard to determine the need for a new regressor without fixing the problem of the condition number.


# LinearAlgebra.cond(θ)
# function lremade!(du,u,p,t)
#     du[1] = -10*u[1] + 10*u[2]
#     du[2] = 15.53 - 15.86*u[1] + 11.26*u[2] + 0.84*u[3]
#     du[3] = -15.63 + 6.99*u[1] + 3.63*u[2] - 3.799*u[3]
# end
#
# p = [10,12,8/3]
# u₀ = [1,0,0]
# tspan = (0,100)
# t = 0:0.01:100
# remade =  ODEProblem(lremade!,u₀,tspan,p)
# sol1 = solve(remade,Tsit5(),saveat=t)
# plot(sol1,vars=(1,2,3))
# savefig("./examples/Learned.png")
#
# obj = LorenzSystem()
# sol = datagen(obj)
# savefig("./examples/Original.png")
