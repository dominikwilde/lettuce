from lettuce.unit import UnitConversion
from lettuce.boundary import BounceBackBoundary

import numpy as np

class TurbulentChannelFlow3D:
    def __init__(self, resolution, reynolds_number, u_tau, lattice):
        mach_number_of_u_tau = np.sqrt(3) * u_tau
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number_of_u_tau,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def initial_solution(self, x):
        # Initialization inspired by a 3D Taylor-Green vortex
        u=np.array([
            np.sin(x[0]) * np.cos(2*x[1]) * np.cos(x[2])*0.05*self.units.characteristic_velocity_pu*16+self.units.characteristic_velocity_pu*16,
            -np.cos(x[0]) * np.sin(x[1]) * np.cos(x[2])*self.units.characteristic_velocity_pu*16,
            np.sin(2*x[2])*self.units.characteristic_velocity_pu*16*0.05
        ])
        p = np.array([0*x[0] + 0 * x[1] + 0*x[2]])
        return p, u

    @property
    def grid(self):
        bounceback_offset = 1./(2*self.resolution)
        x = np.linspace(0, 4*np.pi, num=self.resolution*4*np.pi, endpoint=False)
        # Three additional grid points are needed in y direction, one for the endpoint (via endpoint = True) and two for the bounceback boundaries.
        # The bounceback nodes are located outside the physical domain in the wall (Can be reduced by the use of Half-way bounce back)
        y = np.linspace(0-bounceback_offset, 2+bounceback_offset, num=self.resolution*2+2, endpoint=True)
        z = np.linspace(0, 2*np.pi, num=self.resolution*2*np.pi, endpoint=False)
        return np.meshgrid(x, y, z,indexing='ij')

    @property
    def boundaries(self):
        x,y,z = self.grid
        return [BounceBackBoundary(y < 0.0, self.units.lattice),BounceBackBoundary(y > 2.0, self.units.lattice)]

    def calculate_acceleration(self,Re_tau,visc_lb,resolution):
        # According to Dorschner's PhD thesis:
        acceleration = Re_tau*Re_tau*visc_lb*visc_lb/(resolution*resolution*resolution)
        return acceleration