import numpy as np
import torch

from lettuce.unit import UnitConversion
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary#, ZeroGradientOutletRight, ZeroGradientOutletTop, \
    #ZeroGradientOutletBottom


class Obstacle2D(object):

    def __init__(self, resolution_x, resolution_y, reynolds_number, mach_number, lattice, char_length_lu):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.mask = None

    def initizialize_object(self, mask, lattice):
        self.boundaries.append(BounceBackBoundary(mask, lattice))

    def initial_solution(self, x):
        return np.array([0 * x[0]], dtype=float), np.array(
            [0 * x[0] + self.units.convert_velocity_to_lu(1.0), x[1]*0+self.units.convert_velocity_to_lu(0.05)],
            dtype=float)

    def getMaxU(self, f, lattice):
        u0 = (lattice.u(f)[0])
        u1 = (lattice.u(f)[1])
        return torch.max(torch.sqrt(u0*u0+u1*u1))

    def calcEnstrophy(self, f, lattice):
        u0 = lattice.u(f)[0].cpu().numpy()
        u1 = lattice.u(f)[1].cpu().numpy()
        dx = self.units.convert_length_to_pu(1.0)
        grad_u0 = np.gradient(u0,dx)
        grad_u1 = np.gradient(u1,dx)
        vorticity = np.sum((grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0]))
        vorticity *= dx ** lattice.D
        return  vorticity


    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution_y, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid
        return [EquilibriumBoundaryPU(np.abs(x) < 1e-6, self.units.lattice, self.units, np.array(
            [self.units.characteristic_velocity_pu, self.units.characteristic_velocity_pu * 0.0])),
                ZeroGradientOutletRight(np.abs(y-1) < 1e-6, self.units.lattice),
                #ZeroGradientOutletBottom(np.abs(x-1) < 1e-3, self.units.lattice),
                #ZeroGradientOutletTop(np.abs(x) < 1e-3, self.units.lattice),
                BounceBackBoundary(self.mask, self.units.lattice)]


class ZeroGradientOutletRight:
    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice

    def __call__(self, f):
        f[3, -1] = f[3, -2]
        f[6, -1] = f[6, -2]
        f[7, -1] = f[7, -2]
        return f

class ZeroGradientOutletTop:
    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice

    def __call__(self, f):
        f[2, :, 0] = f[2, :, 1]
        f[5, :, 0] = f[5, :, 1]
        f[6, :, 0] = f[6, :, 1]
        return f

class ZeroGradientOutletBottom:
    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice

    def __call__(self, f):
        f[4, :, -1] = f[4, :, -2]
        f[7, :, -1] = f[7, :, -2]
        f[8, :, -1] = f[8, :, -2]
        return f

class MaxUReporter:
    def __init__(self, lattice, flow, starting_iteration=1000, interval=50):
        self.lattice = lattice
        self.flow = flow
        self.starting_iteration = starting_iteration
        self.interval = interval
        self.out = []

    def __call__(self, i, t, f):
        if t % self.interval == 0:
            if t > self.starting_iteration:
                u0 = (self.lattice.u(f)[0])
                u1 = (self.lattice.u(f)[1])
                max= torch.max(torch.sqrt(u0*u0+u1*u1)).cpu().numpy()
                self.out.append([max])