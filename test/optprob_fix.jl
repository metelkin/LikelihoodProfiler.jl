using Optimization
using ForwardDiff
using LikelihoodProfiler, Plots
using OrdinaryDiffEq

# objective function
rosenbrock(x,p) = (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2

# initial values
x0 = zeros(2)

# solving optimization problem
optf = OptimizationFunction(rosenbrock, AutoForwardDiff())
optprob = OptimizationProblem(optf, x0)
sol = solve(optprob, Optimization.LBFGS())

# optimal values of the parameters
optpars = sol.u

# profile likelihood problem
plprob = PLProblem(optprob, optpars, (-10.,10.); threshold = 4.0)

method = OptimizationProfiler(optimizer = Optimization.LBFGS(), stepper = FixedStep(; initial_step=0.15))
sol = profile(plprob, method)
plot(sol, size=(800,350), margin=5Plots.mm)

method = IntegrationProfiler(integrator = Tsit5(), integrator_opts = (dtmax=0.3,), matrix_type = :hessian)
sol = profile(plprob, method)
plot(sol, size=(800,350), margin=5Plots.mm)