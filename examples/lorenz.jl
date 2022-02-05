using DifferentialEquations
using DataInterpolations
using Plots
using ForwardDiff
using FiniteDiff

#===============================================================#
function datagen()
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
    tspan = (0,10)
    t = 0:0.01:10
    prob = ODEProblem(lorenz!,u₀,tspan,p)
    sol = solve(prob,Tsit5(),saveat=t)
    # noisy = noise(sol.u)
    fig = plot(sol,vars=(1,2,3))
    display(fig)
    return sol
end

#===============================================================#

function denoise!(sol)

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

function differentiate(sol)
    # interp = LagrangeInterpolation(munge(sol.u),sol.t)
    interp = QuadraticInterpolation(munge(sol.u),sol.t)
    y_reconstruct = interp.(sol.t)
    v = [DataInterpolations.derivative(interp,k) for k in sol.t]
    temp = munge(v)
    display(plot(temp[1,:],temp[2,:],temp[3,:]))
    return v
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
    n = 1 + 3 + n2_terms(nparams) # You are better than this!
    θ = zeros(n,ntsteps)
    θ[1,:] = ones(ntsteps)
    θ[2:4,:] = linear(X)
    θ[5:end,:] = quadratic(X)
    return θ
end
#===============================================================#
function _optimize(θ,v,λ)

end
#===============================================================#
# Main function

sol = datagen()
denoise!(sol)
v = differentiate(sol)

X = munge(sol.u)
θ = basis(X)
ξ = _optimize(θ,v,0.1)
