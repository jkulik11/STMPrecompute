The interp_halo_data.py file contains the class necessary to calculate energy optimal control relative transfer costs. Variational data is computed and stored once in a user chosen file, then accessed later.

The OrbitVariationalData object takes in the STM and trajectory data along the reference orbit assuming 2^exponent discretized pieces of the orbit.

The precompute_lu function takes in the initial and final times along the orbit to find the STM associated with those times and LU factorize it.

With the output of precompute_lu, one can call solve_bvp_cost_convenience. This solves the relative motion boundary value problem to move the secondary satellite from the initial relative position to the final relative position. It assumes the additional boundary condition that the inertial relative velocities of the two satellites at the beginning and end should be zero. This method returns the energy optimal control cost to satisfy the given boundary conditions.
