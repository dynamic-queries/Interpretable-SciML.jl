include("../src/SINDY.jl")
using LinearAlgebra

# Sample and test(integration tests) some of the functionalities for Types
ds = LorenzSystem()
ds1 = ODE1()
ds2 = ODE3()
opt = STLSQ(0.01)
lib = TrigBasis()
lib1 = PolyTrigBasis()

# Integration tests - lorenz attractor
begin
        data = datagen(LorenzSystem())
        savefig("./figures/Traj_Lorenz.svg")
        v = differentiate(data,TotalVariationalDerivativative())
        savefig("./figures/Vel_Lorenz.svg")
        b_type = PolyTrigBasis()
        θ = basis(munge(data.u),b_type)
        LinearAlgebra.cond(θ)
        ξ = _optimize(θ,v,STLSQ(0.1))
        pprint(ξ,b_type)

        # This function needs to be defined manually from the output of pretty print.
        function lorenz_remake!(du,u,p,t)
                du[1] = -9.9756*u[1] + 9.97594*u[2]
                du[2] = 27.7373*u[1] - 0.9388*u[2] - 0.99*u[1]*u[3]
                du[3] = -2.65769*u[3] +  0.99615*u[1]*u[2]
        end

        u₀ = [1,0,0]
        p = [10,28,8/3]
        tspan = (0,100)
        t = 0.01:0.01:100

        prob = ODEProblem(lorenz_remake!,u₀,tspan,p)
        sol = solve(prob,Tsit5(),saveat=t)
        fig = plot(sol,vars=(1,2,3),title="Lorenz Attractor remade")
        display(fig)
        savefig("./figures/Remade_Lorenz.svg")
end


#TODO: The basis function is the only manual part of the implementation. This has to change.
#TODO: Include the forcing function expressions in the defn of the Dynamical System.

# Integration tests - Lotka Volterra

begin
        data = datagen(LotkaVolterra())
        savefig("./figures/Traj_Lotka.svg")
        v = differentiate(data,TotalVariationalDerivativative())
        savefig("./figures/Vel_Lotka.svg")
        b_type = PolynomialBasis()
        θ = basis(munge(data.u),b_type)
        LinearAlgebra.cond(θ)
        ξ = _optimize(θ,v,STLSQ(0.01))
        pprint(ξ,b_type)


        function lk!(du,u,p,t)
                du[1] = 0.7003*u[1] - 0.30014*u[1]*u[2]
                du[2] = -0.299953*u[2] + 0.399979*u[1]*u[2]
        end

        u₀ = [1,1]
        p = []
        tspan = (0,20)
        t = 0.01:0.01:20
        prob = ODEProblem(lk!,u₀,tspan,p)
        sol = solve(prob,Tsit5(),saveat=t)
        fig = plot(sol,vars=(1,2),title="Lotka Volterra remade")
        display(fig)
        savefig("./figures/Remade_Lotka.svg")
end
