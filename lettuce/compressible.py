import numpy as np
from lettuce import Stencil, UnitConversion, Lattice, Simulation, LettuceException, BounceBackBoundary
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

def makeD2Q25HWeights():
    r = (np.sqrt(5.) - np.sqrt(2.)) / np.sqrt(3.)
    w_0 = (-3 - 3 * r * r * r * r + 54 * r * r) / (75 * r * r);
    w_m = (9 * r * r * r * r - 6 - 27 * r * r) / (300 * r * r * (r * r - 1))
    w_n = (9 - 6 * r * r * r * r - 27 * r * r) / (300 * (1 - r * r))
    w_0n = w_0 * w_n
    w_0m = w_0 * w_m
    w_mm = w_m * w_m
    w_mn = w_m * w_n
    w_nn = w_n * w_n
    result = []
    result += w_0 * w_0, w_0m, w_0m, w_0m, w_0m, w_mm, w_mm, w_mm, w_mm, w_0n, w_0n, w_0n, w_0n, w_nn, w_nn, w_nn, w_nn, w_mn, w_mn, w_mn, w_mn, w_mn, w_mn, w_mn, w_mn
    w = np.asarray(result)
    return w

def makeD2Q25HSpeeds():
    c_m = np.sqrt(5. - np.sqrt(10.))  # /np.sqrt(3.)
    c_n = np.sqrt(5. + np.sqrt(10.))  # /np.sqrt(3.)
    e = [[0.0, 0.0], [c_m, 0.0], [0.0, c_m], [-c_m, 0.0], [0.0, -c_m], [c_m, c_m], [-c_m, c_m], [-c_m, -c_m],
         [c_m, -c_m], [c_n, 0.0], [0.0, c_n], [-c_n, 0.0], [0.0, -c_n], [c_n, c_n], [-c_n, c_n], [-c_n, -c_n],
         [c_n, -c_n], [c_m, c_n], [c_m, -c_n], [-c_m, -c_n], [-c_m, c_n], [c_n, c_m], [c_n, -c_m], [-c_n, -c_m],
         [-c_n, c_m]]
    e = np.asarray(e)
    return e

class D2Q25H(Stencil):

    e = makeD2Q25HSpeeds()
    w = makeD2Q25HWeights()

    cs = 1

class HermiteEquilibrium:
    def __init__(self, lattice):
        self.lattice = lattice
        e = lattice.e
        cs2 = lattice.cs*lattice.cs
        Q = e.shape[0]
        D = e.shape[1]

        #Calculate Hermite polynomials
        self.H0=1
        self.H1 = torch.zeros([Q, D],dtype=lattice.dtype)
        self.H2 = torch.zeros([Q, D, D],dtype=lattice.dtype)
        self.H3 = torch.zeros([Q, D, D, D],dtype=lattice.dtype)
        self.H4 = torch.zeros([Q, D, D, D, D],dtype=lattice.dtype)

        for a in range(D):
            self.H1[:,a]= e[:,a]
            for b in range(D):
                e_ab =  e[:,a]*e[:,b]
                self.H2[:, a, b] = e_ab
                if(a==b):
                    self.H2[:, a, b] -= cs2
                for c in range(D):
                    e_abc = e_ab*e[:,c]
                    self.H3[:,a,b,c] = e_abc
                    if a == b:
                        self.H3[:, a, b, c] -= e[:, c] * cs2
                    if b == c:
                        self.H3[:, a, b, c] -= e[:, a] * cs2
                    if a == c:
                        self.H3[:, a, b, c] -= e[:, b] * cs2
                    for d in range(D):
                        self.H4[:,a,b,c,d] = e_abc * e[:,d]
                        if a == b:
                            if c == d:
                                self.H4[:, a, b, c, d] += cs2 * cs2
                        if b == c:
                            if a == d:
                                self.H4[:, a, b, c, d] += cs2 * cs2
                        if b == d:
                            if a == c:
                                self.H4[:, a, b, c, d] += cs2 * cs2
                        if a == b:
                            self.H4[:, a, b, c, d] -= cs2 * e[:, c] * e[:, d]
                        if a == c:
                            self.H4[:, a, b, c, d] -= cs2 * e[:, b] * e[:, d]
                        if a == d:
                            self.H4[:, a, b, c, d] -= cs2 * e[:, b] * e[:, c]
                        if b == c:
                            self.H4[:, a, b, c, d] -= cs2 * e[:, a] * e[:, d]
                        if b == d:
                            self.H4[:, a, b, c, d] -= cs2 * e[:, a] * e[:, c]
                        if c == d:
                            self.H4[:, a, b, c, d] -= cs2 * e[:, a] * e[:, b]

    def __call__(self, rho, u, T, *args):
        D = self.lattice.e.shape[1]
        eye=torch.eye(2)

        cs2 = self.lattice.cs*self.lattice.cs
        T_1 = (T - 1) * cs2
        a0 = 1
        a1 = u
        a2 = self.lattice.einsum('a,b->ab',[a1,a1]) + torch.eye(2)*T_1
        a3 = self.lattice.einsum('ab,c->abc',[self.lattice.einsum('a,b->ab',[a1,a1]),a1])


        for c in range(D):
            a3[:, :, c] += T_1 * eye * u[c]
            a3[:, c, :] += T_1 * eye * u[c]
            a3[c, :, :] += T_1 * eye * u[c]

        a4 = self.lattice.einsum('abc,d->abcd',[self.lattice.einsum('ab,c->abc',[self.lattice.einsum('a,b->ab',[a1,a1]),a1]),a1])
        for a in range(D):
            for b in range(D):
                for c in range(D):
                    for d in range(D):
                        if (a==b):
                            a4[a,b,c,d] += u[c]*u[d]*T_1
                        if (a==c):
                            a4[a,b,c,d] += u[b]*u[d]*T_1
                        if (a==d):
                            a4[a,b,c,d] += u[c]*u[b]*T_1
                        if (b==c):
                            a4[a,b,c,d] += u[a]*u[d]*T_1
                        if (b==d):
                            a4[a,b,c,d] += u[a]*u[c]*T_1
                        if (c==d):
                            a4[a,b,c,d] += u[a]*u[b]*T_1
                        if a == b:
                            if c == d:
                                a4[a, b, c, d] += T_1*T_1
                        if b == c:
                            if a == d:
                                a4[a, b, c, d] += T_1*T_1
                        if b == d:
                            if a == c:
                                a4[a, b, c, d] += T_1*T_1


        H0a0 = self.H0*a0
        H1a1 = self.lattice.einsum('ia,a->i',[self.H1,a1])
        H2a2 = self.lattice.einsum('iab,ab->i',[self.H2,a2])
        H3a3 = self.lattice.einsum('iabc,abc->i',[self.H3,a3])
        H4a4 = self.lattice.einsum('iabcd,abcd->i', [self.H4, a4])

        feq = self.lattice.w * rho * (
                    H0a0 + H1a1 / (cs2) + H2a2 / (2 * cs2 * cs2) + H3a3 / (6 * cs2 * cs2 * cs2)
        + H4a4 / (24 * cs2 * cs2 * cs2 * cs2))
        print(self.H4[1])
        return feq

class SodShockTube:
    def __init__(self, resolution, lattice, T_left=1, rho_left=8):
        self.resolution = resolution
        self.T_left = T_left
        self.rho_left = rho_left
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
        rho[:,0:int(self.resolution/2),] = 8
        return rho, u, T

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution, endpoint=False)
        y = np.linspace(0, 1, num=5, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid
        return [BounceBackBoundary(np.abs(x) < 1e-6, self.units.lattice),BounceBackBoundary(np.abs(x) -1  < 1e-6, self.units.lattice)]


class CompressibleLattice(Lattice):
    def temperature(self, f):
        u = self.u(f)
        diff = (self.e - u)*(self.e - u)
        prod = torch.sum(diff, axis=1)
        return self.einsum("i,i->", [prod, f])  / (self.rho(f)*self.cs * self.cs * 2)


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

        self.f = lattice.equilibrium(lattice.convert_to_tensor(rho), lattice.convert_to_tensor(u))

        self.reporters = []

        # Define a mask, where the collision shall not be applied
        x = flow.grid
        self.no_collision_mask = np.zeros_like(x[0], dtype=bool)
        self.no_collision_mask = lattice.convert_to_tensor(self.no_collision_mask)
        for boundary in self.flow.boundaries:
            if boundary.__class__.__name__ == "BounceBackBoundary":
                self.no_collision_mask = boundary.mask | self.no_collision_mask


