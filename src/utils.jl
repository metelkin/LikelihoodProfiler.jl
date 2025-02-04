
"""
    chi2_quantile(α, df=1)
Computes α quantile for Chi2 distribution with df degrees of freedom
"""
chi2_quantile(α, df=1) = quantile(Chisq(df), α)

compute_optf(prob::OptimizationProblem, u, p=prob.p) = compute_optf(prob.f, u, p)
compute_optf(f::OptimizationFunction, u, p=SciMLBase.NullParameters()) = f(u, p)
#compute_optf(f::GetindexFunction, u) = f(u)

function compute_grad(prob::OptimizationProblem, u, p=prob.p)
  if !isnothing(prob.f.grad)
    return prob.f.grad(u,p)
  elseif prob.f.adtype isa SciMLBase.NoAD
    # f = Base.Fix2(prob.f.f, prob.p)
    DifferentiationInterface.gradient(x->prob.f(x,p), prob.f.adtype, u)
  else
    grad_error()
  end
end

function compute_hess(prob::OptimizationProblem, u, p=prob.p)
  if !isnothing(prob.f.hess)
    return prob.f.hess(u_modified!,p)
  elseif prob.f.adtype isa SciMLBase.NoAD
    DifferentiationInterface.hessian(x->prob.f(x,p), prob.f.adtype, u)
  else
    hess_error()  
  end
end

grad_error() = throw(ArgumentError("Gradient is not provided. Use `OptimizationFunction` either with a valid AD backend https://docs.sciml.ai/Optimization/stable/API/ad/ or a provided 'grad' function."))
hess_error() = throw(ArgumentError("Hessian is not provided. Use `OptimizationFunction` either with a valid AD backend https://docs.sciml.ai/Optimization/stable/API/ad/ or a provided 'hess' function."))


# tmp fix to interpret correctly NLopt retcodes
function deduce_retcode(retcode::Symbol)
  if retcode == :Default || retcode == :DEFAULT
      return ReturnCode.Default
  elseif retcode == :Success || retcode == :EXACT_SOLUTION_LEFT ||
         retcode == :FLOATING_POINT_LIMIT || retcode == :true || retcode == :OPTIMAL ||
         retcode == :LOCALLY_SOLVED || retcode == :ROUNDOFF_LIMITED || retcode == :SUCCESS ||
        
         # tmp fix to return correct status
         retcode == :SUCCESS || retcode == :STOPVAL_REACHED || retcode == :FTOL_REACHED || retcode == :XTOL_REACHED
      
      return ReturnCode.Success
  elseif retcode == :Terminated
      return ReturnCode.Terminated
  elseif retcode == :MaxIters || retcode == :MAXITERS_EXCEED ||
         retcode == :MAXEVAL_REACHED
      return ReturnCode.MaxIters
  elseif retcode == :MaxTime || retcode == :TIME_LIMIT
      return ReturnCode.MaxTime
  elseif retcode == :DtLessThanMin
      return ReturnCode.DtLessThanMin
  elseif retcode == :Unstable
      return ReturnCode.Unstable
  elseif retcode == :InitialFailure
      return ReturnCode.InitialFailure
  elseif retcode == :ConvergenceFailure || retcode == :ITERATION_LIMIT
      return ReturnCode.ConvergenceFailure
  elseif retcode == :Failure || retcode == :false ||

      # tmp fix to return correct status
      retcode == :FAILURE || retcode == :OUT_OF_MEMORY || retcode == :INVALID_ARGS || retcode == :FORCED_STOP
      
      return ReturnCode.Failure
  elseif retcode == :Infeasible || retcode == :INFEASIBLE ||
         retcode == :DUAL_INFEASIBLE || retcode == :LOCALLY_INFEASIBLE ||
         retcode == :INFEASIBLE_OR_UNBOUNDED
      return ReturnCode.Infeasible
  else
      return ReturnCode.Failure
  end
end


# helper type used in parameter profiling
struct FixedParamCache{P}
  p::P
  idx::Base.RefValue{Int}
  x_fixed::Base.RefValue{Float64}
end

function FixedParamCache(p, idx::Int, x_fixed::Real)
  _p = isnothing(p) ? SciMLBase.NullParameters() : p
  return FixedParamCache{typeof(_p)}(_p, Ref(idx), Ref(float(x_fixed)))
end

get_idx(p::FixedParamCache) = p.idx[]
get_x_fixed(p::FixedParamCache) = p.x_fixed[]
set_idx!(p::FixedParamCache, idx::Int) = p.idx[] = idx
set_x_fixed!(p::FixedParamCache, x_fixed::Float64) = p.x_fixed[] = x_fixed

fill_x_full!(x_full::AbstractVector{T}, x_reduced::AbstractVector{T}, p::FixedParamCache) where T<:Number = 
  fill_x_full!(x_full, x_reduced, get_idx(p), get_x_fixed(p))

function fill_x_full!(x_full::AbstractVector{T}, x_reduced::AbstractVector{T}, idx::Int, x_fixed) where T<:Number
  for i in 1:idx-1
    x_full[i] = x_reduced[i]
  end
  x_full[idx] = T(x_fixed)
  for i in idx+1:length(x_full)
    x_full[i] = x_reduced[i-1]
  end
end

function fill_x_reduced!(x_reduced::AbstractVector{T}, x_full::AbstractVector{T}, idx::Int) where T
  for i in 1:idx-1
    x_reduced[i] = x_full[i]
  end
  for i in idx+1:length(x_full)
    x_reduced[i-1] = x_full[i]
  end
end

# fill elements of reduced Matrix
function fill_x_reduced!(x_reduced::AbstractMatrix{T}, x_full::AbstractMatrix{T}, idx::Int) where T
  nr, nc = size(x_full)
  for i in 1:idx-1
    for j in 1:idx-1
      x_reduced[i, j] = x_full[i, j]
    end
    for j in idx+1:nc
      x_reduced[i, j-1] = x_full[i, j]
    end
  end

  for i in idx+1:nr
    for j in 1:idx-1
      x_reduced[i-1, j] = x_full[i, j]
    end
    for j in idx+1:nc
      x_reduced[i-1, j-1] = x_full[i, j]
    end
  end
end

get_solver_stats(solver_state::SciMLBase.AbstractODEIntegrator) = solver_state.sol.stats
get_solver_stats(solver_state::SciMLBase.AbstractOptimizationCache) = SciMLBase.OptimizationStats()

