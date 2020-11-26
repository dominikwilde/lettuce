"""
Boundary Conditions.

The `__call__` function of a boundary defines its application to the distribution functions.

Boundary conditions can define a mask (a boolean numpy array)
that specifies the grid points on which the boundary
condition operates.

Boundary classes can define two functions `make_no_stream_mask` and `make_no_collision_mask`
that prevent streaming and collisions on the boundary nodes.

The no-stream mask has the same dimensions as the distribution functions (Q, x, y, (z)) .
The no-collision mask has the same dimensions as the grid (x, y, (z)).

"""

import torch
import numpy as np
from lettuce import (LettuceException)


__all__ = ["BounceBackBoundary", "AntiBounceBackOutlet", "EquilibriumBoundaryPU", "EquilibriumOutletP",
           "ZeroGradientOutlet", "BounceBackVelocityInlet", "EquilibriumExtrapolationOutlet",
           "NonEquilibriumExtrapolationOutlet", "NonEquilibriumExtrapolationInletU"]


class DirectionalBoundary(object):
    pass


class ObjectBoundary(object):
    pass


class BounceBackBoundary:
    """Fullway Bounce-Back Boundary"""
    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice

        self.output_force = False
        self.force = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))
        if lattice.D == 2:
            x, y = mask.shape
            self.force_mask = np.zeros((lattice.Q, x, y), dtype=bool)
            a, b = np.where(mask)
            for p in range(0, len(a)):
                for i in range(0, lattice.Q):
                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1]]:
                            self.force_mask[self.lattice.stencil.opposite[i], a[p], b[p]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if lattice.D == 3:
            x, y, z = mask.shape
            self.force_mask = np.zeros((lattice.Q, x, y, z), dtype=bool)
            a, b, c = np.where(mask)
            for p in range(0, len(a)):
                for i in range(0, lattice.Q):
                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1], c[p] + lattice.stencil.e[i, 2]]:
                            self.force_mask[self.lattice.stencil.opposite[i], a[p], b[p], c[p]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        self.force_mask = self.lattice.convert_to_tensor(self.force_mask)

    def __call__(self, f, index=...):
        if self.output_force:
            tmp = torch.where(self.force_mask, f, torch.zeros_like(f))
            self.force = 1 ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / 1.0
            #tmp = torch.einsum("i..., id -> d...", tmp, self.lattice.e)
            #for _ in range(0, self.lattice.D):
            #    tmp = torch.sum(tmp, dim=1)
            #self.force = tmp * 2
        f = torch.where(self.mask[index], f[self.lattice.stencil.opposite], f)
        return f

    def make_no_collision_mask(self, grid_shape):
        #assert self.mask.shape == grid_shape[0]
        return self.mask


class EquilibriumBoundaryPU:
    """Sets distributions on this boundary to equilibrium with predefined velocity and pressure.
    Note that this behavior is generally not compatible with the Navier-Stokes equations.
    This boundary condition should only be used if no better options are available.
    """
    def __init__(self, mask, lattice, units, velocity, pressure=0):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.units = units
        self.velocity = lattice.convert_to_tensor(velocity)
        self.pressure = lattice.convert_to_tensor(pressure)

    def __call__(self, f):
        rho = self.units.convert_pressure_pu_to_density_lu(self.pressure)
        u = self.units.convert_velocity_to_lu(self.velocity)
        feq = self.lattice.equilibrium(rho, u)
        feq = self.lattice.einsum("q,q->q", [feq, torch.ones_like(f)])
        f = torch.where(self.mask, feq, f)
        return f


class AntiBounceBackOutlet:
    """Allows distributions to leave domain unobstructed through this boundary.
        Based on equations from page 195 of "The lattice Boltzmann method" (2016 by Krüger et al.)
        Give the side of the domain with the boundary as list [x, y, z] with only one entry nonzero
        [1, 0, 0] for positive x-direction in 3D; [1, 0] for the same in 2D
        [0, -1, 0] is negative y-direction in 3D; [0, -1] for the same in 2D
        """

    def __init__(self, lattice, direction):
        assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
            LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
                                f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        direction = np.array(direction)
        self.lattice = lattice

        #select velocities to be bounced (the ones pointing in "direction")
        self.velocities = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, direction) > 1 - 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)
        # construct indices for einsum and get w in proper shape for the calculation in each dimension
        if len(direction) == 3:
            self.dims = 'dc, cxy -> dxy'
            self.w = self.lattice.w[self.velocities].view(1, -1).t().unsqueeze(1)
        if len(direction) == 2:
            self.dims = 'dc, cx -> dx'
            self.w = self.lattice.w[self.velocities].view(1, -1).t()
        if len(direction) == 1:
            self.dims = 'dc, c -> dc'
            self.w = self.lattice.w[self.velocities]

    def __call__(self, f):
        u = self.lattice.u(f)
        u_w = u[[slice(None)] + self.index] + 0.5 * (u[[slice(None)] + self.index] - u[[slice(None)] + self.neighbor])
        f[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = (
            - f[[self.velocities] + self.index] + self.w * self.lattice.rho(f)[[slice(None)] + self.index] *
            (2 + torch.einsum(self.dims, self.lattice.e[self.velocities], u_w) ** 2 / self.lattice.cs ** 4
             - (torch.norm(u_w,dim=0) / self.lattice.cs) ** 2)
        )
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = 1
        return no_stream_mask

    # not 100% sure about this. But collisions seem to stabilize the boundary.
    #def make_no_collision_mask(self, f_shape):
    #    no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
    #    no_collision_mask[self.index] = 1
    #    return no_collision_mask


class EquilibriumOutletP(AntiBounceBackOutlet):
    """Equilibrium outlet with constant pressure.
    """
    def __init__(self, lattice, direction, rho0=1.0):
        super(EquilibriumOutletP, self).__init__(lattice, direction)
        self.rho0 = rho0

    def __call__(self, f):
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        rho_w = self.rho0 * torch.ones_like(rho[here])
        u_w = u[other]
        f[here] = self.lattice.equilibrium(rho_w[...,None], u_w[...,None])[...,0]
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.setdiff1d(np.arange(self.lattice.Q), self.velocities)] + self.index] = 1
        return no_stream_mask

    def make_no_collision_mask(self, f_shape):
        no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
        no_collision_mask[self.index] = 1
        return no_collision_mask

class EquilibriumExtrapolationOutlet(AntiBounceBackOutlet):
    """Equilibrium outlet with extrapolated pressure and velocity from inside the domain
    """
    def __init__(self, lattice, direction):
        super(EquilibriumExtrapolationOutlet, self).__init__(lattice, direction)

    def __call__(self, f):
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        rho_w = rho[other] #+ 0.5 * (rho[here] - rho[other])
        u_w = u[other] #+ 0.5 * (u[here] - u[other])
        f[here] = self.lattice.equilibrium(rho_w[...,None], u_w[...,None])[...,0]
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.setdiff1d(np.arange(self.lattice.Q), self.velocities)] + self.index] = 1
        return no_stream_mask

    def make_no_collision_mask(self, grid_shape):
        no_collision_mask = torch.zeros(size=grid_shape, dtype=torch.bool, device=self.lattice.device)
        no_collision_mask[self.index] = 1
        return no_collision_mask

class ZeroGradientOutlet(object):

    def __init__(self, lattice, direction):
        assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
            LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
                                f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        self.direction = np.array(direction)
        self.lattice = lattice

        #select velocities to be replaced (the ones pointing against "direction")
        self.velocities = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) < -1 + 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in self.direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)

    def __call__(self, f):
        f[[self.velocities] + self.index] = f[[self.velocities] + self.neighbor]
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[self.velocities] + self.index] = 1
        return no_stream_mask


class BounceBackVelocityInlet(object):
    """Allows distributions to enter domain with set speed through this boundary.
        Based on page 195 of "The lattice Boltzmann method" (2016 by Krüger et al.)
        Give the side of the domain with the boundary as list [x, y, z] with only one entry nonzero
        [1, 0, 0] for positive x-direction in 3D; [1, 0] for the same in 2D
        [0, -1, 0] is negative y-direction in 3D; [0, -1] for the same in 2D
        """

    def __init__(self, lattice, units, direction, velocity_pu):
        assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
            LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
                                f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        self.direction = np.array(direction)
        self.lattice = lattice
        self.units = units
        self.velocity_lu = units.convert_velocity_to_lu(self.lattice.convert_to_tensor(velocity_pu))

        #select velocities to be bounced (the ones pointing in "direction")
        self.velocities_out = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) > 1 - 1e-6), axis=0)
        # select velocities to be replaced (the ones pointing against "direction")
        self.velocities_in = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) < -1 + 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in self.direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)

    def __call__(self, f):
        rho = self.lattice.rho(f[[slice(None)] + self.neighbor])
        rho_w = torch.mean(rho) #rho[[slice(None)] + self.index] + 0.5 * (rho[[slice(None)] + self.index] - rho[[slice(None)] + self.neighbor])  # extrapolation of rho_w from density at boundary and neighbour node, hopefully better than global average / 1
        f[[self.velocities_in] + self.index] = (
                f[[self.velocities_out] + self.index] - 2 * rho_w * (self.lattice.w[self.velocities_out] * torch.matmul(self.lattice.e[self.velocities_out], self.velocity_lu) / self.lattice.cs ** 2).view(3, 1)
        )
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[self.velocities_in] + self.index] = 1
        return no_stream_mask

class NonEquilibriumExtrapolationOutlet(object):
    """ Zou's boundary condition
    use on post stream populations!!!!!
        """

    def __init__(self, lattice, rho_w, direction):
        assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
            LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
                                f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        self.direction = np.array(direction)
        self.lattice = lattice
        self.rho_w = self.lattice.convert_to_tensor(rho_w)

        # select velocities to be bounced (the ones pointing in "direction")
        self.velocities_out = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) > 1 - 1e-6), axis=0)
        # select velocities to be replaced (the ones pointing against "direction")
        self.velocities_in = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) < -1 + 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in self.direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)

    def __call__(self, f):
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        u = self.lattice.convert_to_tensor(self.lattice.u(f[other + [None]]))
        u_w = u.detach().clone()
        # smart (?) methode to find velocity orthogonal to boundary... doesnt help for parallel velocity though
        u_w[np.argwhere(self.direction != 0)[0][0]] = self.lattice.e[self.velocities_in[0], np.argwhere(self.direction != 0)[0][0]] * (1 - 1 / self.rho_w * (
            torch.sum(f[[np.setdiff1d(np.arange(self.lattice.Q), [self.velocities_in, self.velocities_out])] + self.index]
                      + 2 * f[[self.velocities_out] + self.index], dim=0))).unsqueeze(1)
        rho = self.lattice.convert_to_tensor(self.lattice.rho(f[other + [None]]))
        rho_w = self.rho_w * torch.ones_like(rho)
        f[here] = self.lattice.equilibrium(rho_w, u_w).squeeze(2) + (f[other] - self.lattice.equilibrium(rho, u).squeeze(2))
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[self.velocities_in] + self.index] = 1
        return no_stream_mask

class NonEquilibriumExtrapolationInletU(object):
    """ Zou's boundary condition
    use on post stream populations!!!!!
        """

    def __init__(self, lattice, u_w, direction):
        assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
            LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
                                f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        self.direction = np.array(direction)
        self.lattice = lattice
        self.u_w = self.lattice.convert_to_tensor(u_w)

        # select velocities to be bounced (the ones pointing in "direction")
        self.velocities_out = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) > 1 - 1e-6), axis=0)
        # select velocities to be replaced (the ones pointing against "direction")
        self.velocities_in = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) < -1 + 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in self.direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)
        self.rho_old = 1.0

    def __call__(self, f):
        Tc = 10
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        u = self.lattice.convert_to_tensor(self.lattice.u(f[other + [None]]))
        rho = self.lattice.convert_to_tensor(self.lattice.rho(f[other + [None]]))
        u_w = self.u_w.unsqueeze(1).repeat(1, f.shape[np.argwhere(self.direction == 0)[0][0] + 1]).unsqueeze(2)
        rho_self = 1 / (1 - u_w[np.argwhere(self.direction != 0)[0][0]] / self.lattice.e[self.velocities_in[0], np.argwhere(self.direction != 0)[0][0]]) * (
            torch.sum(f[[np.setdiff1d(np.arange(self.lattice.Q), [self.velocities_in, self.velocities_out])] + self.index] + 2 * f[[self.velocities_out] + self.index], dim=0)).unsqueeze(1)
        # desnity filtering as proposed by https://www.researchgate.net/publication/257389374_Computational_Gas_Dynamics_with_the_Lattice_Boltzmann_Method_Preconditioning_and_Boundary_Conditions
        rho_w = (rho_self + Tc * self.rho_old) / (1+Tc)
        self.rho_old = rho_w
        f[here] = self.lattice.equilibrium(rho_w, u_w).squeeze(2) + (f[other] - self.lattice.equilibrium(rho, u).squeeze(2))
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[self.velocities_in] + self.index] = 1
        return no_stream_mask

