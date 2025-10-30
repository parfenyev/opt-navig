## Content

This folder contains scripts that are used to generate data for 2D turbulence example.

`./get_data/script.jl` -- The script is used to integrate the Navier-Stokes equation from rest, and after a transient process the system reaches the statistical steady-state.

`./get_data/script2.jl` -- Now the system is already in the statistical steady-state and we are generating the data for 2d turbulence example.

`./processing.ipynb` -- Data visualization, velocity calculation based on stored vorticity using periodic boundary conditions and energy spectrum construction.

All scripts in this folder are written in [Julia](https://julialang.org/). Don't forget to update the file paths in the scripts. You can also [download the generated data](https://parfenyev.itp.ac.ru/data/opt-navig/) directly.