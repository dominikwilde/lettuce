rho_0 = 1.0

import torch
from lettuce import Simulation, Guo, UnitConversion
from timeit import default_timer as timer
import numpy as np

def psi_function(lattice,f):
    psi = torch.zeros_like(f)
    psi[:] = rho_0 * (1.0 - torch.exp(-lattice.rho(f)/rho_0))
    return psi

def shan_chen_forcing_magnitude(lattice,psi,G=1.1):
    psi_c = torch.einsum('i...,ia->ia...',psi[1:],lattice.e[1:])

    F = - psi[0] * G * torch.einsum('i,ia...->a...',lattice.w[1:],psi_c)
    return F

class MultiphaseSimulation(Simulation):

    def step(self,num_steps):
        """Take num_steps stream-and-collision steps and return performance in MLUPS."""
        start = timer()
        for _ in range(num_steps):
            self.i += 1
            self.f = self.streaming(self.f)
            #Perform the collision routine everywhere, expect where the no_collision_mask is true
            psi = psi_function(self.lattice,self.f)
            psi = self.streaming(psi)
            F=shan_chen_forcing_magnitude(self.lattice,psi)
            force = Guo(self.lattice,self.collision.tau,F)
            self.f = torch.where(self.no_collision_mask, self.f, self.collision(self.f,force))
            for boundary in self.flow.boundaries:
                self.f = boundary(self.f)
            for reporter in self.reporters:
                reporter(self.i, self.i, self.f)
        end = timer()
        seconds = end-start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds
        return mlups


class SimpleMultiphaseFlow:
    def __init__(self, resolution, lattice, reynolds_number, mach_number):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1)


    def analytic_solution(self, x, t=0):
        raise NotImplementedError

    def initial_solution(self, x):

        ux = x[0] * 0.0 + x[1] * 0.0
        uy = x[0] * 0.0 + x[1] * 0.0

        u = np.array([ux, uy])
        p = np.array([np.zeros(x[0].shape)])
        if (x[0]>0.25 and x[0]<0.75 and x[1]>0.25 and x[1]<0.75 ):
            p=3.0

        return p, u

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y)


    @property
    def boundaries(self):
        return []