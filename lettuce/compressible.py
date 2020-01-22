import numpy as np
from lettuce import Stencil, UnitConversion, Lattice, Simulation, LettuceException
import torch


class D2Q37(Stencil):
    e = np.array(
        [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1], [2, 0], [-2, 0], [0, 2], [0, -2],
         [2, 1], [-2, -1], [2, -1], [-2, 1], [1, 2], [-1, -2], [1, -2], [-1, 2], [2, 2], [-2, -2], [2, -2], [-2, 2],
         [3, 0], [-3, 0], [0, 3], [0, -3], [3, 1], [-3, -1], [3, -1], [-3, 1], [1, 3], [-1, -3], [1, -3], [-1, 3]])
    w = np.array(
        [0.233151] + [0.107306] * 4 + [0.0576679] * 4 + [0.0142082] * 4 + [0.00535305] * 8 + [0.00101194] * 4 + [
            0.000245301] * 4 + [0.000283414] * 8)
    cs = 1.196979 #1. / 6 * np.sqrt(49 - (119 + (469 + 252 * np.sqrt(30)) ** (2. / 3)) / ((469 + 252 * np.sqrt(30)) ** (1. / 3.)))  #
    opposite = [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13,
        16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31, 34, 33, 36, 35
    ]


class SodShockTube:
    def __init__(self, resolution, lattice, dimension):
        self.resolution = resolution
        self.reynolds_number = 1
        self.units = UnitConversion(
            lattice,
            reynolds_number=1, mach_number=1,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )


    def initial_solution(self, x):
        u = np.array([0 * x[0] + 0 * x[1], 0 * x[0] + 0 * x[1]])
        rho = np.array([x[0] * 0 + x[1] * 0 + 0.8])
        T = np.array([x[0] * 0 + x[1] * 0 + 0.8])
        rho[:,0:int(self.resolution/2),] = 1.25
        return rho, u, T

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution, endpoint=False)
        y = np.linspace(0, 1, num=5, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        return []


class CompressibleLattice(Lattice):
    def temperature(self, f):
        u = self.u(f)
        diff = self.e - u
        prod = torch.prod(diff, axis=1) / (self.cs * self.cs2)
        return self.einsum("i,i->", prod, f)


class CompressibleSimulation(Simulation):
    def __init__(self, flow, lattice, collision, streaming):
        self.flow = flow
        self.lattice = lattice
        self.collision = collision
        self.streaming = streaming
        self.i = 0

        grid = flow.grid
        rho, u, T = flow.initial_solution(grid)
        assert list(rho.shape) == [1] + list(grid[0].shape), \
            LettuceException(f"Wrong dimension of initial pressure field. "
                             f"Expected {[1] + list(grid[0].shape)}, "
                             f"but got {list(p.shape)}.")
        assert list(u.shape) == [lattice.D] + list(grid[0].shape), \
            LettuceException("Wrong dimension of initial velocity field."
                             f"Expected {[lattice.D] + list(grid[0].shape)}, "
                             f"but got {list(u.shape)}.")
        #u = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
        #rho = lattice.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(p))
        self.f = lattice.equilibrium(lattice.convert_to_tensor(rho), lattice.convert_to_tensor(u))

        self.reporters = []

        # Define a mask, where the collision shall not be applied
        x = flow.grid
        self.no_collision_mask = np.zeros_like(x[0], dtype=bool)
        self.no_collision_mask = lattice.convert_to_tensor(self.no_collision_mask)
        for boundary in self.flow.boundaries:
            if boundary.__class__.__name__ == "BounceBackBoundary":
                self.no_collision_mask = boundary.mask | self.no_collision_mask


test = D2Q37()
print(test.cs)
