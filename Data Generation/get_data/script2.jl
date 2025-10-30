using GeophysicalFlows, Random, Printf, CUDA, JLD2
using LinearAlgebra: mul!, ldiv!
parsevalsum = FourierFlows.parsevalsum

#CUDA.device!(3)
dev = GPU()     # Device (CPU/GPU)

# ## Numerical, domain, and simulation parameters

 n, L  = 512, 2π              # grid resolution and domain length
 ν, nν = 0.005, 1             # viscosity coefficient and hyperviscosity order
 μ, nμ = 0.2, 0              # linear drag coefficient
    dt = 0.0002                # timestep
nsteps = 30750            # total number of steps
 nsubs = 25                # number of steps between each plot

# ## Forcing

forcing_wavenumber = 5.5 * 2π/L  # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
forcing_bandwidth  = 1.0  * 2π/L  # the width of the forcing spectrum, `δ_f`
ε = 10.0                           # energy input rate by the forcing

grid = TwoDGrid(dev; nx=n, Lx=L)

K = @. sqrt(grid.Krsq)             # a 2D array with the total wavenumber

forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
@CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average

ε0 = parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε/ε0        # normalize forcing to inject energy at rate ε

random_uniform = dev==CPU() ? rand : CUDA.rand

function calcF!(Fh, sol, t, clock, vars, params, grid) 
  Fh .= sqrt.(forcing_spectrum) .* exp.(2π * im * random_uniform(eltype(grid), size(sol))) ./ sqrt(clock.dt)
  
  return nothing
end

# ## Problem setup

prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt, stepper="ETDRK4",
                                calcF=calcF!, stochastic=true)

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y

calcF!(vars.Fh, sol, 0.0, clock, vars, params, grid)

# ## Setting initial conditions

dev2 = CPU()
grid2 = TwoDGrid(dev2; nx=n, Lx=L)

data = jldopen("/storage/p2/parfenyev/2d_turbulence/RL/alpha=0.2/alpha_0.2_v1.jld2")
in_cond_h = data["snapshots/sol/50000"]
in_cond = zeros(Int(n),Int(n))
ldiv!(in_cond, grid2.rfftplan, deepcopy(in_cond_h))
close(data)

TwoDNavierStokes.set_ζ!(prob, device_array(dev)(in_cond))

# ## Diagnostics

E  = Diagnostic(TwoDNavierStokes.energy, prob, nsteps=nsteps) # energy
Z = Diagnostic(TwoDNavierStokes.enstrophy, prob; nsteps=nsteps)
W = Diagnostic(TwoDNavierStokes.energy_work, prob, nsteps=nsteps) # energy work input by forcing
diags = [E, Z, W] # a list of Diagnostics passed to `stepforward!` will  be updated every timestep.

# ## Output

filepath = "/storage/p2/parfenyev/2d_turbulence/RL/alpha=0.2"
filename = joinpath(filepath, "alpha_0.2_v2.jld2")
if isfile(filename); rm(filename); end

get_sol(prob) = Array(prob.sol) # extracts variables
out = Output(prob, filename, (:sol, get_sol), 
    (:E, TwoDNavierStokes.energy), (:Z, TwoDNavierStokes.enstrophy), (:W, TwoDNavierStokes.energy_work))
saveproblem(out)


# Finally, we time-step the `Problem` forward in time.

startwalltime = time()

for j = 0:round(Int, nsteps / nsubs)
  cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
  log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Z: %.4f, W: %.4f, walltime: %.2f min",
        clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], W.data[W.i], (time()-startwalltime)/60)
  println(log)
  
  stepforward!(prob, diags, nsubs)
  TwoDNavierStokes.updatevars!(prob)
  saveoutput(out)
end