import numpy as np
from lettuce import UnitConversion, Stencil, write_vtk, UnitConversion, VTKReporter, torch_gradient,Lattice, Simulation, LettuceException, BounceBackBoundary, GenericStepReporter
import torch
from timeit import default_timer as timer


class D2Q37(Stencil):
    e = np.array(
        [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1], [2, 0], [-2, 0], [0, 2], [0, -2],
         [2, 1], [-2, -1], [2, -1], [-2, 1], [1, 2], [-1, -2], [1, -2], [-1, 2], [2, 2], [-2, -2], [2, -2], [-2, 2],
         [3, 0], [-3, 0], [0, 3], [0, -3], [3, 1], [-3, -1], [3, -1], [-3, 1], [1, 3], [-1, -3], [1, -3], [-1, 3]])
    w = np.array(
        [0.233151] + [0.107306] * 4 + [0.0576679] * 4 + [0.0142082] * 4 + [0.00535305] * 8 + [0.00101194] * 4 + [
            0.000245301] * 4 + [0.000283414] * 8)
    cs = 1/1.1969797752#0.83546599  # 1.1969797752 #1. / 6 * np.sqrt(49 - (119 + (469 + 252 * np.sqrt(30)) ** (2. / 3)) / ((469 + 252
    # * np.sqrt(30)) ** (1. / 3.)))  #
    opposite = [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13,
        16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31, 34, 33, 36, 35
    ]

class D3V59(Stencil):
    e = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 1], [0, 1, -1],
         [0, -1, -1], [1, 0, -1], [0, 1, 1], [1, 0, 1], [-1, 0, 1], [-1, 1, 0], [-1, 0, -1], [-1, -1, 0], [1, -1, 0],
         [1, 1, 0], [-1, 1, -1], [-1, 1, 1], [-1, -1, 1], [-1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1], [1, -1, -1],
         [0, 0, 2], [0, 2, 0], [2, 0, 0], [0, 0, -2], [0, -2, 0], [-2, 0, 0], [-2, -2, 0], [-2, 0, 2], [-2, 0, -2],
         [2, 2, 0], [0, 2, -2], [0, -2, -2], [0, 2, 2], [2, -2, 0], [2, 0, -2], [-2, 2, 0], [2, 0, 2], [0, -2, 2],
         [-3, 0, 0], [0, 0, 3], [0, 0, -3], [0, -3, 0], [0, 3, 0], [3, 0, 0], [2, 2, 2], [2, -2, -2], [2, -2, 2],
         [-2, 2, 2], [-2, 2, -2], [-2, -2, 2], [2, 2, -2], [-2, -2, -2]])
    weights =[]
    weights+=[9.58789162377528*10**(-2)]*1
    weights+=[7.31047082129148*10**(-2)]*6
    weights+=[3.46588971093380*10**(-3)]*12
    weights+=[3.66108082044515*10**(-2)]*8
    weights+=[1.59235232232060*10**(-2)]*6
    weights+=[2.52480845105094*10**(-3)]*12
    weights+=[7.26968662515159*10**(-5)]*8
    weights+=[7.65879439346840*10**(-4)]*6
    w = np.array(weights)
    cs = 1/1.20288512331026

class D3V107(Stencil):
    e = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, 0, -1], [1, 0, 0], [0, -1, 0], [1, -1, 0], [-1, 1, 0],
         [0, 1, 1], [-1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, -1, 0], [0, -1, -1], [1, 1, 0], [1, 0, 1], [0, -1, 1],
         [0, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, 1, -1],
         [0, -2, 0], [-2, 0, 0], [0, 2, 0], [0, 0, -2], [0, 0, 2], [2, 0, 0], [0, 2, -1], [0, -2, 1], [0, 2, 1],
         [0, -1, 2], [0, -1, -2], [0, 1, -2], [0, -2, -1], [-1, 0, 2], [1, -2, 0], [2, 1, 0], [2, 0, 1], [2, 0, -1],
         [2, -1, 0], [-2, -1, 0], [-2, 0, -1], [-2, 0, 1], [0, 1, 2], [-2, 1, 0], [1, 0, 2], [-1, -2, 0], [-1, 0, -2],
         [1, 0, -2], [-1, 2, 0], [1, 2, 0], [0, 2, 2], [2, 0, -2], [0, 2, -2], [2, 0, 2], [2, -2, 0], [2, 2, 0],
         [0, -2, 2], [0, -2, -2], [-2, -2, 0], [-2, 0, -2], [-2, 0, 2], [-2, 2, 0], [3, 1, 1], [-3, -1, -1],
         [-3, -1, 1], [-3, 1, -1], [-3, 1, 1], [1, 3, 1], [1, 3, -1], [1, 1, 3], [3, -1, -1], [-1, -3, -1], [1, 1, -3],
         [-1, -1, -3], [1, -3, -1], [1, -3, 1], [3, 1, -1], [-1, -3, 1], [-1, 3, 1], [-1, 3, -1], [1, -1, -3],
         [1, -1, 3], [-1, 1, -3], [-1, -1, 3], [-1, 1, 3], [3, -1, 1], [2, -2, -2], [-2, 2, -2], [2, -2, 2],
         [-2, -2, 2], [-2, -2, -2], [-2, 2, 2], [2, 2, 2], [2, 2, -2], [-4, 0, 0], [0, 0, 4], [0, 0, -4], [0, -4, 0],
         [0, 4, 0], [4, 0, 0]])
    weights = []
    weights += [7.57516860965017 * 10 ** (-2)] * 1
    weights += [6.00912802747447 * 10 ** (-2)] * 6
    weights += [3.13606906699535 * 10 ** (-3)] * 12
    weights += [3.63392812078012 * 10 ** (-2)] * 8
    weights += [1.32169332731492 * 10 ** (-2)] * 6
    weights += [4.48492851172950 * 10 ** (-3)] * 24
    weights += [2.48755775808342 * 10 ** (-3)] * 12
    weights += [6.07432754970149 * 10 ** (-4)] * 24
    weights += [4.64179164402822 * 10 ** (-4)] * 8
    weights += [4.51928894609872 * 10 ** (-5)] * 6
    w = np.array(weights)

    cs = 1/1.07182071542885

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

class CompressibleUnitConversion(UnitConversion):
    @property
    def characteristic_velocity_lu(self):
        return self.lattice.stencil.cs * self.mach_number * np.sqrt(self.lattice.gamma)


class HermiteEquilibrium:
    def __init__(self, lattice):
        self.lattice = lattice
        e = lattice.e
        cs2 = lattice.cs * lattice.cs
        Q = e.shape[0]
        D = e.shape[1]

        # Calculate Hermite polynomials
        self.H0 = 1
        self.H1 = torch.zeros([Q, D], dtype=lattice.dtype,device=lattice.device)
        self.H2 = torch.zeros([Q, D, D], dtype=lattice.dtype,device=lattice.device)
        self.H3 = torch.zeros([Q, D, D, D], dtype=lattice.dtype,device=lattice.device)
        self.H4 = torch.zeros([Q, D, D, D, D], dtype=lattice.dtype,device=lattice.device)

        for a in range(D):
            self.H1[:, a] = e[:, a]
            for b in range(D):
                e_ab = e[:, a] * e[:, b]
                self.H2[:, a, b] = e_ab
                if (a == b):
                    self.H2[:, a, b] -= cs2
                for c in range(D):
                    e_abc = e_ab * e[:, c]
                    self.H3[:, a, b, c] = e_abc
                    if a == b:
                        self.H3[:, a, b, c] -= e[:, c] * cs2
                    if b == c:
                        self.H3[:, a, b, c] -= e[:, a] * cs2
                    if a == c:
                        self.H3[:, a, b, c] -= e[:, b] * cs2
                    for d in range(D):
                        self.H4[:, a, b, c, d] = e_abc * e[:, d]
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

    def __call__(self, rho, u, T, order=4, *args):
        D = self.lattice.e.shape[1]

        cs2 = self.lattice.cs * self.lattice.cs
        T_1 = ((T - 1) * cs2)[0]
        a0 = 1
        a1 = u
        a2 = self.lattice.einsum('a,b->ab', [a1, a1]) #+ torch.eye(2) * T_1
        for a in range(D):
            for b in range(D):
                if a==b:
                    a2[a,b]+=T_1
        a3 = self.lattice.einsum('ab,c->abc', [self.lattice.einsum('a,b->ab', [a1, a1]), a1])

        for a in range(D):
            for b in range(D):
                for c in range(D):
                    if a==b:
                        a3[a, b, c] += T_1 * u[c]
                    if b==c:
                        a3[a, b, c] += T_1 * u[a]
                    if a==c:
                        a3[a, b, c] += T_1 * u[b]

        a4 = self.lattice.einsum('abc,d->abcd',
                                 [self.lattice.einsum('ab,c->abc', [self.lattice.einsum('a,b->ab', [a1, a1]), a1]), a1])
        for a in range(D):
            for b in range(D):
                for c in range(D):
                    for d in range(D):
                        if (a == b):
                            a4[a, b, c, d] += u[c] * u[d] * T_1
                        if (a == c):
                            a4[a, b, c, d] += u[b] * u[d] * T_1
                        if (a == d):
                            a4[a, b, c, d] += u[c] * u[b] * T_1
                        if (b == c):
                            a4[a, b, c, d] += u[a] * u[d] * T_1
                        if (b == d):
                            a4[a, b, c, d] += u[a] * u[c] * T_1
                        if (c == d):
                            a4[a, b, c, d] += u[a] * u[b] * T_1
                        if a == b:
                            if c == d:
                                a4[a, b, c, d] += T_1 * T_1
                        if b == c:
                            if a == d:
                                a4[a, b, c, d] += T_1 * T_1
                        if b == d:
                            if a == c:
                                a4[a, b, c, d] += T_1 * T_1

        H0a0 = self.H0 * a0
        H1a1 = self.lattice.einsum('ia,a->i', [self.H1, a1])
        H2a2 = self.lattice.einsum('iab,ab->i', [self.H2, a2])
        H3a3 = self.lattice.einsum('iabc,abc->i', [self.H3, a3])
        H4a4 = self.lattice.einsum('iabcd,abcd->i', [self.H4, a4])

        feq = rho*self.lattice.einsum('i,i->i', [self.lattice.w,(
                H0a0 + H1a1 / (cs2) + H2a2 / (2 * cs2 * cs2) + H3a3 / (6 * cs2 * cs2 * cs2)
                + H4a4 / (24 * cs2 * cs2 * cs2 * cs2))])
        return feq


class SodShockTube:
    def __init__(self, resolution, lattice, T_left=1.25, rho_left=8, visc=0.001):
        self.visc = visc
        self.resolution = resolution
        self.T_left = T_left
        self.rho_left = rho_left
        self.units = UnitConversion(
            lattice,
            reynolds_number=lattice.cs/(visc*resolution), mach_number=1,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def calc_offsets(self, left, right):
        d = (right / 2 - left / 2)
        c = 2 * (left) / (right - left) + 1
        return c, d

    def initial_solution(self, x):
        #Defines the smoothness of the discontinuity
        a=10000000000
        rho_c,rho_d=self.calc_offsets(self.rho_left,1.0)
        T_c,T_d = self.calc_offsets(self.T_left,1.0)
        u = np.array([0 * x[0] + 0 * x[1], 0 * x[0] + 0 * x[1]])
        rho = np.array([(np.tanh(a*(x[0]-0.5))+rho_c)*rho_d+ x[1] * 0 ])
        T = np.array([(np.tanh(a*(x[0]-0.5))+T_c)*T_d + x[1] * 0])
        return rho, u, T

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution, endpoint=True)
        y = np.linspace(0, 1, num=5, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid
        return [BounceBackBoundary(np.abs(x) < 1e-1, self.units.lattice),BounceBackBoundary(np.abs(x) > 9*1e-1, self.units.lattice)]

class SteadyShock:
    def __init__(self, resolution, lattice, T_left=1.25, rho_left=8, visc=0.001):
        self.visc = visc
        self.resolution = resolution
        self.T_left = T_left
        self.rho_left = rho_left
        self.units = UnitConversion(
            lattice,
            reynolds_number=lattice.cs/(visc*resolution), mach_number=1,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def calc_offsets(self, left, right):
        d = (right / 2 - left / 2)
        c = 2 * (left) / (right - left) + 1
        return c, d

    def initial_solution(self, x):
        #Defines the smoothness of the discontinuity
        a=10000000000
        rho_c,rho_d=self.calc_offsets(self.rho_left,1.0)
        T_c,T_d = self.calc_offsets(self.T_left,1.0)
        u = np.array([0 * x[0] + 0 * x[1], 0 * x[0] + 0 * x[1]])
        rho = np.array([(np.tanh(a*(x[0]-0.5))+rho_c)*rho_d+ x[1] * 0 ])
        T = np.array([(np.tanh(a*(x[0]-0.5))+T_c)*T_d + x[1] * 0])
        return rho, u, T

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution, endpoint=True)
        y = np.linspace(0, 1, num=5, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid
        return [BounceBackBoundary(np.abs(x) < 1e-1, self.units.lattice),BounceBackBoundary(np.abs(x) > 9*1e-1, self.units.lattice)]

class CompressibleEnergyReporter(GenericStepReporter):
    """Reports the kinetic energy """
    _parameter_name = 'Kinetic energy'

    def parameter_function(self,i,t,f,g):
        dx = self.flow.units.convert_length_to_pu(1.0)
        gamma = 1.4
        C_v = 1. / (gamma - 1)
        T = self.lattice.temperature(f,g,C_v)
        rho = self.lattice.rho(f)
        p = rho*T
        kinE = self.flow.units.convert_incompressible_energy_to_pu(torch.sum(self.lattice.incompressible_energy(f)*rho ))
        kinE *= dx ** self.lattice.D
        return kinE.item()

class MaxMachReporter(GenericStepReporter):
    """Reports the maximum Mach number"""
    _parameter_name = 'Max Mach number'

    def parameter_function(self, i, t, f, g):
        gamma = 1.4
        C_v = 1. / (gamma - 1)
        u = self.lattice.u(f)
        magnitude = u[0]**2 + u[1]**2
        if(self.lattice.D==3):
            magnitude += u[2]**2
        magnitude = torch.sqrt(magnitude)

        T = self.lattice.temperature(f,g,C_v)
        #a_0 = torch.sqrt(gamma*T)
        magnitude = torch.max(magnitude/(self.lattice.cs*np.sqrt(1.4))) #/ a_0)
        return magnitude



class CompressibleDissipationReporter(GenericStepReporter):
    """Reports the integral of the vorticity

    Notes
    -----
    The function only works for periodic domains
    """
    _parameter_name = 'Dissipation'

    def parameter_function(self,i,t,f,g):
        u0 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[0])
        u1 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[1])
        dx = self.flow.units.convert_length_to_pu(1.0)
        grad_u0 = torch_gradient(u0, dx=dx, order=6).cpu().numpy()
        grad_u1 = torch_gradient(u1, dx=dx, order=6).cpu().numpy()
        dilatation = np.sum(grad_u0[0] + grad_u1[1])
        vorticity = np.sum((grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0]))
        if self.lattice.D == 3:
            u2 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[2])
            grad_u2 = torch_gradient(u2, dx=dx, order=6).cpu().numpy()
            dilatation += np.sum(grad_u2[2])
            vorticity += np.sum(
                (grad_u2[1] - grad_u1[2]) * (grad_u2[1] - grad_u1[2])
                + ((grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0])))
        return (vorticity+4./3*dilatation.item()) * dx**self.lattice.D

class CompressibleLattice(Lattice):
    def __init__(self, stencil, device, dtype=torch.float, gamma = 1.4):
        super().__init__(stencil,device,dtype)
        self.gamma = gamma

    def temperature(self, f, g, C_v):
        u = self.u(f)
        index = [Ellipsis] + [None] * self.D
        diff = (self.e[index] - u ) * (self.e[index] - u)
        prod = torch.sum(diff, axis=1)
        f_sum = (self.einsum("i,i->i", [prod, f]))
        fg_sum = (f_sum/(self.cs * self.cs) + g).sum(axis=0)
        return  fg_sum/ (self.rho(f) * 2*C_v)


class BGKCompressibleCollision:
    def __init__(self, lattice, tau, gamma):
        self.lattice = lattice
        self.tau = tau
        self.gamma = gamma

    def __call__(self, f, g):

        C_v = 1./(self.gamma - 1)
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        T = self.lattice.temperature(f,g,C_v)
        feq = self.lattice.equilibrium(rho, u, T)
        sunderland_factor = 1.4042 * T**1.5 / (T+0.40417)
        sunderland_tau = (self.tau -0.5)*sunderland_factor + 0.5
        geq = (2*C_v-self.lattice.D)*T*feq
        f_post= f - 1.0 / sunderland_tau * (f-feq)
        g_post = g - 1.0 / sunderland_tau *(g-geq)
        return f_post,g_post

class CompressibleVTKReporter(VTKReporter):
    def __call__(self, i, t, f, g):
        if t % self.interval == 0:
            t = self.flow.units.convert_time_to_pu(t)
            u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
            rho = self.lattice.rho(f)
            T = self.lattice.temperature(f,g,2.5)
            if self.lattice.D == 2:
                self.point_dict["p"] = self.lattice.convert_to_numpy(rho[0, ..., None])
                for d in range(self.lattice.D):
                    self.point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ..., None])
                self.point_dict["T"] = self.lattice.convert_to_numpy(T[0, ...])
            else:
                self.point_dict["p"] = self.lattice.convert_to_numpy(rho[0, ...])
                for d in range(self.lattice.D):
                    self.point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ...])
            self.point_dict["T"] = self.lattice.convert_to_numpy(T[0, ...])
            write_vtk(self.point_dict, i, self.filename_base)

class CompressibleSimulation(Simulation):
    def __init__(self, flow, lattice, collision, streaming):
        self.flow = flow
        self.lattice = lattice
        self.collision = collision
        self.streaming = streaming
        self.i = 0
        self.lattice.equilibrium=HermiteEquilibrium(lattice)
        self.reporters = []

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
        u = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
        self.f = lattice.equilibrium(lattice.convert_to_tensor(rho), lattice.convert_to_tensor(u),lattice.convert_to_tensor(T))
        gamma = 1.4
        C_v = 1/(gamma - 1)
        self.g = (2*C_v-lattice.D)*lattice.convert_to_tensor(T)*self.f


        # Define a mask, where the collision shall not be applied
        x = flow.grid
        self.no_collision_mask = np.zeros_like(x[0], dtype=bool)
        self.no_collision_mask = lattice.convert_to_tensor(self.no_collision_mask)
        for boundary in self.flow.boundaries:
            if boundary.__class__.__name__ == "BounceBackBoundary":
                self.no_collision_mask = boundary.mask | self.no_collision_mask

    def step(self, num_steps):
        """Take num_steps stream-and-collision steps and return performance in MLUPS."""
        start = timer()
        for _ in range(num_steps):
            self.i += 1
            self.f = self.streaming(self.f)
            self.g = self.streaming(self.g)


            f_post, g_post = self.collision(self.f,self.g)
                # Perform the collision routine everywhere, expect where the no_collision_mask is true
            self.f = torch.where(self.no_collision_mask, self.f, f_post)
            self.g = torch.where(self.no_collision_mask, self.g, g_post)
            for boundary in self.flow.boundaries:
                self.f = boundary(self.f)
                self.g = boundary(self.g)

            for reporter in self.reporters:
                reporter(self.i, self.i, self.f, self.g)

        end = timer()
        seconds = end - start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds
        return mlups

    def initialize_f_neq(self):
        """Initialize the distribution function values. The f^(1) contributions are approximated by finite differences.
        See Krüger et al. (2017).
        """
        rho = self.lattice.rho(self.f)
        u = self.lattice.u(self.f)
        T = self.lattice.temperature(self.f,self.g,2.5)

        grad_u0 = torch_gradient(u[0], dx=1, order=6)[None, ...]
        grad_u1 = torch_gradient(u[1], dx=1, order=6)[None, ...]
        S = torch.cat([grad_u1, grad_u0])

        if(self.lattice.D==3):
            grad_u2 = torch_gradient(u[2], dx=1, order=6)[None, ...]
            S = torch.cat([S, grad_u2])

        Pi_1 = -1.0 * self.flow.units.relaxation_parameter_lu * rho  * S / self.lattice.cs**2
        Q = torch.einsum('ia,ib->iab',self.lattice.e,self.lattice.e)-torch.eye(self.lattice.D,device=self.lattice.device)*self.lattice.cs**2
        Pi_1_Q = torch.einsum('ab...,iab->i...', Pi_1, Q)
        fneq = torch.einsum('i,i...->i...',self.lattice.w,Pi_1_Q)

        feq = self.lattice.equilibrium(rho,u,T)
        self.f = feq + fneq


