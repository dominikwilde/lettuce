"""
Moments and cumulants of the distribution function.
"""

import warnings
import torch
import lettuce
from lettuce.util import LettuceException, InefficientCodeWarning, get_subclasses, ExperimentalWarning
from lettuce.stencils import *
from lettuce.equilibrium import Equilibrium


_ALL_STENCILS = get_subclasses(Stencil, module=lettuce)


def moment_tensor(e, multiindex):
    if isinstance(e, torch.Tensor):
        return torch.prod(torch.pow(e, multiindex[..., None, :]), dim=-1)
    else:
        return np.prod(np.power(e, multiindex[..., None, :]), axis=-1)


def get_default_moment_transform(lattice):
    if lattice.stencil == D1Q3:
        return D1Q3Transform(lattice)
    if lattice.stencil == D2Q9:
        return D2Q9Lallemand(lattice)
    else:
        raise LettuceException(f"No default moment transform for lattice {lattice}.")


class Moments:
    def __init__(self, lattice):
        self.rho = moment_tensor(lattice.e, lattice.convert_to_tensor(np.zeros(lattice.D)))
        self.j = moment_tensor(lattice.e, lattice.convert_to_tensor(np.eye(lattice.D)))
        # ... TODO ...


class Transform:
    """Base class that defines the signature for all moment (and cumulant) transforms.
    """
    def __init__(self, lattice, names=None):
        self.lattice = lattice
        self.names = [f"m{i}" for i in range(lattice.Q)] if names is None else names

    def __getitem__(self, moment_names):
        if not isinstance(moment_names, tuple):
            moment_names = [moment_names]
        return [self.names.index(name) for name in moment_names]

    def transform(self, f):
        return f

    def inverse_transform(self, m):
        return m

    def equilibrium(self, m):
        """A very inefficient and basic implementation of the equilibrium moments.
        """
        warnings.warn(
            "Transform.equilibrium is a poor man's implementation of the moment equilibrium."
            "Please consider implementing the equilibrium moments for your transform by hand.",
            InefficientCodeWarning
        )
        f = self.inverse_transform(m)
        feq = self.lattice.equilibrium(self.lattice.rho(f), self.lattice.u(f))
        return self.transform(feq)


class D1Q3Transform(Transform):
    matrix = np.array([
        [1,1,1],
        [0,1,-1],
        [0,1,1]
    ])
    inverse = np.array([
        [1, 0, -1],
        [0, 1/2, 1/2],
        [0, -1/2, 1/2]
    ])
    names = ["rho", "j", "e"]
    supported_stencils = [D1Q3]

    def __init__(self, lattice):
        super(D1Q3Transform, self).__init__(lattice, self.names)
        self.matrix = self.lattice.convert_to_tensor(self.matrix)
        self.inverse = self.lattice.convert_to_tensor(self.inverse)

    def transform(self, f):
        return self.lattice.mv(self.matrix, f)

    def inverse_transform(self, m):
        return self.lattice.mv(self.inverse, m)

    #def equilibrium(self, m):
    #    # TODO
    #    raise NotImplementedError


class D2Q9Dellar(Transform):

    matrix = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 0, -1, 0, 1, -1, -1, 1],
         [0, 0, 1, 0, -1, 1, 1, -1, -1],
         [-3 / 2, 3, -3 / 2, 3, -3 / 2, 3, 3, 3, 3],
         [0, 0, 0, 0, 0, 9, -9, 9, -9],
         [-3 / 2, -3 / 2, 3, -3 / 2, 3, 3, 3, 3, 3],
         [1, -2, -2, -2, -2, 4, 4, 4, 4],
         [0, -2, 0, 2, 0, 4, -4, -4, 4],
         [0, 0, -2, 0, 2, 4, 4, -4, -4]]
    )
    inverse = np.array(
        [[4 / 9, 0, 0, -4 / 27, 0, -4 / 27, 1 / 9, 0, 0],
         [1 / 9, 1 / 3, 0, 2 / 27, 0, -1 / 27, -1 / 18, -1 / 12, 0],
         [1 / 9, 0, 1 / 3, -1 / 27, 0, 2 / 27, -1 / 18, 0, -1 / 12],
         [1 / 9, -1 / 3, 0, 2 / 27, 0, -1 / 27, -1 / 18, 1 / 12, 0],
         [1 / 9, 0, -1 / 3, -1 / 27, 0, 2 / 27, -1 / 18, 0, 1 / 12],
         [1 / 36, 1 / 12, 1 / 12, 1 / 54, 1 / 36, 1 / 54, 1 / 36, 1 / 24, 1 / 24],
         [1 / 36, -1 / 12, 1 / 12, 1 / 54, -1 / 36, 1 / 54, 1 / 36, -1 / 24, 1 / 24],
         [1 / 36, -1 / 12, -1 / 12, 1 / 54, 1 / 36, 1 / 54, 1 / 36, -1 / 24, -1 / 24],
         [1 / 36, 1 / 12, -1 / 12, 1 / 54, -1 / 36, 1 / 54, 1 / 36, 1 / 24, -1 / 24]]
    )
    names = ['rho', 'jx', 'jy', 'Pi_xx', 'Pi_xy', 'PI_yy', 'N', 'Jx', 'Jy']
    supported_stencils = [D2Q9]

    def __init__(self, lattice):
        super(D2Q9Dellar, self).__init__(
            lattice, self.names
        )
        self.matrix = self.lattice.convert_to_tensor(self.matrix)
        self.inverse = self.lattice.convert_to_tensor(self.inverse)

    def transform(self, f):
        return self.lattice.mv(self.matrix, f)

    def inverse_transform(self, m):
        return self.lattice.mv(self.inverse, m)

    def equilibrium(self, m):
        warnings.warn("I am not 100% sure if this equilibrium is correct.", ExperimentalWarning)
        meq = torch.zeros_like(m)
        rho = m[0]
        jx = m[1]
        jy = m[2]
        Pi_xx = jx*jx/rho*9/2
        Pi_xy = jx*jy/rho*9
        Pi_yy = jy*jy/rho*9/2
        meq[0] = rho
        meq[1] = jx
        meq[2] = jy
        meq[3] = Pi_xx
        meq[4] = Pi_xy
        meq[5] = Pi_yy
        return meq


class D2Q9Lallemand(Transform):

    matrix = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 0, -1, 0, 1, -1, -1, 1],
         [0, 0, 1, 0, -1, 1, 1, -1, -1],
         [0, 1, -1, 1, -1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, -1, 1, -1],
         [-4, -1, -1, -1, -1, 2, 2, 2, 2],
         [0, -2, 0, 2, 0, 1, -1, -1, 1],
         [0, 0, -2, 0, 2, 1, 1, -1, -1],
         [4, -2, -2, -2, -2, 1, 1, 1, 1]]
    )
    inverse = np.array(
        [[1 / 9, 0, 0, 0, 0, -1 / 9, 0, 0, 1 / 9],
         [1 / 9, 1 / 6, 0, 1 / 4, 0, -1 / 36, -1 / 6, 0, -1 / 18],
         [1 / 9, 0, 1 / 6, -1 / 4, 0, -1 / 36, 0, -1 / 6, -1 / 18],
         [1 / 9, -1 / 6, 0, 1 / 4, 0, -1 / 36, 1 / 6, 0, -1 / 18],
         [1 / 9, 0, -1 / 6, -1 / 4, 0, -1 / 36, 0, 1 / 6, -1 / 18],
         [1 / 9, 1 / 6, 1 / 6, 0, 1 / 4, 1 / 18, 1 / 12, 1 / 12, 1 / 36],
         [1 / 9, -1 / 6, 1 / 6, 0, -1 / 4, 1 / 18, -1 / 12, 1 / 12, 1 / 36],
         [1 / 9, -1 / 6, -1 / 6, 0, 1 / 4, 1 / 18, -1 / 12, -1 / 12, 1 / 36],
         [1 / 9, 1 / 6, -1 / 6, 0, -1 / 4, 1 / 18, 1 / 12, -1 / 12, 1 / 36]]
    )
    names = ['rho', 'jx', 'jy', 'pxx', 'pxy', 'e', 'qx', 'qy', 'eps']
    supported_stencils = [D2Q9]

    def __init__(self, lattice):
        super(D2Q9Lallemand, self).__init__(
            lattice, self.names
        )
        self.matrix = self.lattice.convert_to_tensor(self.matrix)
        self.inverse = self.lattice.convert_to_tensor(self.inverse)

    def transform(self, f):
        return self.lattice.mv(self.matrix, f)

    def inverse_transform(self, m):
        return self.lattice.mv(self.inverse, m)

    def equilibrium(self, m):
        """From Lallemand and Luo"""
        warnings.warn("I am not 100% sure if this equilibrium is correct.", ExperimentalWarning)
        meq = torch.zeros_like(m)
        rho = m[0]
        jx = m[1]
        jy = m[2]
        c1 = -2
        alpha2 = -8
        alpha3 = 4
        gamma1 = 2/3
        gamma2 = 18
        gamma3 = 2/3
        gamma4 = -18
        e = 1/4*alpha2*rho+1/6*gamma2*(jx**2 + jy**2)
        eps = 1/4*alpha3*rho + 1/6*gamma4*(jx**2+jy**2)
        qx = 1/2*c1*jx
        qy = 1/2*c1*jy
        pxx = 1/2*gamma1*(jx**2-jy**2)
        pxy = 1/2*gamma3*(jx*jy)
        meq[0] = rho
        meq[1] = jx
        meq[2] = jy
        meq[3] = pxx
        meq[4] = pxy
        meq[5] = e
        meq[6] = qx
        meq[7] = qy
        meq[8] = eps
        return meq


"""
D3Q19 is not implemented, yet. Also, the moments should be ordered so that 1...D+1 correspond to momentum,
which is no the case for this matrix.
"""
# class D3Q19DHumieres(NaturalMomentTransform):
#     matrix = np.array(
#         [[1 / 1, 1, 1, 1, 1, 1, 1, 1, 1 / 1, 1, 1, 1, 1, 1, 1, 1, 1 / 1, 1, 1],
#         [-30, -11, -11, -11 / 1, -11, -11, -11, 8, 8, 8, 8 / 1, 8, 8, 8, 8, 8, 8, 8, 8 / 1],
#         [12, -4, -4, -4, -4, -4 / 1, -4, 1, 1, 1, 1, 1, 1, 1 / 1, 1, 1, 1, 1, 1],
#         [0, 1 / 1, 0, -1, 0, 0, 0, 1, -1, -1, 1, 1, 1, -1, -1, 0, 0, 0, 0],
#         [0, -4, 0, 4, 0, 0, 0, 1 / 1, -1, -1, 1, 1, 1, -1, -1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, -1, 1, 1, -1, -1, 1 / 1, 1, -1],
#         [0, 0, 0, 0, 0, 4, -4, 0, 0, 0, 0, -1, 1, 1, -1, -1, 1, 1, -1 / 1],
#         [0, 0, 1, 0, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, 1, -1, -1],
#         [0, 0, -4, 0, 4, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, 1 / 1, -1, -1],
#         [0, 2, -1, 2 / 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2 / 1],
#         [0, -4, 2, -4, 2, 2 / 1, 2, 1, 1, 1, 1, 1, 1, 1 / 1, 1, -2, -2, -2, -2],
#         [0, 0, -1, 0, -1, 1, 1, -1, -1 / 1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0],
#         [0, 0, 2 / 1, 0, 2, -2, -2, -1, -1, -1 / 1, -1, 1, 1, 1, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, -1, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, -1, 1],
#         [0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, -1 / 1, 1, 1, -1, 1, 1, -1, -1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, -1, 1, 1, -1],
#         [0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, -1. / 1, -1, 1, 1]]
#     )
#     inverse = np.array(
#         [[1 / 19, -5 / 399, 1 / 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [1 / 19, -11 / 2394, -1 / 63, 1 / 10, -1 / 10, 0, 0, 0, 0, 1 / 18, -1 / 18, 0, 0, 0, 0, 0, 0, 0, 0],
#          [1 / 19, -11 / 2394, -1 / 63, 0, 0, 0, 0, 1 / 10, -1 / 10, -1 / 36, 1 / 36, -1 / 12, 1 / 12, 0, 0, 0, 0, 0, 0],
#          [1 / 19, -11 / 2394, -1 / 63, -1 / 10, 1 / 10, 0, 0, 0, 0, 1 / 18, -1 / 18, 0, 0, 0, 0, 0, 0, 0, 0],
#          [1 / 19, -11 / 2394, -1 / 63, 0, 0, 0, 0, -1 / 10, 1 / 10, -1 / 36, 1 / 36, -1 / 12, 1 / 12, 0, 0, 0, 0, 0, 0],
#          [1 / 19, -11 / 2394, -1 / 63, 0, 0, -1. / 10, 1 / 10, 0, 0, -1 / 36, 1 / 36, 1 / 12, -1 / 12, 0, 0, 0, 0, 0, 0],
#          [1 / 19, -11. / 2394, -1 / 63, 0, 0, 1 / 10, -1. / 10, 0, 0, -1 / 36, 1 / 36, 1 / 12, -1 / 12, 0, 0, 0, 0, 0, 0],
#          [1 / 19, 4 / 1197, 1. / 252, 1 / 10, 1 / 40, 0, 0, 1 / 10, 1 / 40, 1 / 36, 1 / 72, -1 / 12, -1 / 24, 0, 0, 1 / 4, -1 / 8, 0, 1. / 8],
#          [1 / 19, 4 / 1197, 1 / 252, -1 / 10, -1 / 40, 0, 0, 1 / 10, 1 / 40, 1 / 36, 1 / 72, -1 / 12, -1 / 24, 0, 0, -1 / 4, 1. / 8, 0, 1 / 8],
#          [1 / 19, 4. / 1197, 1 / 252, -1 / 10, -1 / 40, 0, 0, -1 / 10, -1 / 40, 1 / 36, 1 / 72, -1. / 12, -1 / 24, 0, 0, 1 / 4, 1 / 8, 0, -1 / 8],
#          [1 / 19, 4 / 1197, 1. / 252, 1 / 10, 1 / 40, 0, 0, -1. / 10, -1 / 40, 1 / 36, 1 / 72, -1 / 12, -1. / 24, 0, 0, -1 / 4, -1 / 8, 0, -1 / 8],
#          [1 / 19, 4 / 1197, 1 / 252, 1. / 10, 1 / 40, -1 / 10, -1 / 40, 0, 0, 1 / 36, 1 / 72, 1 / 12, 1 / 24, -1 / 4, 0, 0, 1 / 8, 1 / 8, 0],
#          [1 / 19, 4 / 1197, 1 / 252, 1 / 10, 1 / 40, 1. / 10, 1 / 40, 0, 0, 1 / 36, 1. / 72, 1 / 12, 1 / 24, 1 / 4, 0, 0, 1 / 8, - 1 / 8, 0],
#          [1. / 19, 4 / 1197, 1 / 252, - 1 / 10, - 1 / 40, 1. / 10, 1 / 40, - 0, - 0, 1 / 36, 1 / 72, 1 / 12, 1 / 24, - 1 / 4, 0, 0, -1. / 8, -1 / 8, 0],
#          [1 / 19, 4. / 1197, 1 / 252, -1 / 10, -1 / 40, -1 / 10, -1. / 40, 0, 0, 1 / 36, 1 / 72, 1 / 12, 1 / 24, 1 / 4, 0, 0, -1 / 8, 1 / 8, 0],
#          [1 / 19, 4 / 1197, 1 / 252, 0, 0, -1 / 10, -1 / 40, 1 / 10, 1 / 40, -1 / 18, -1 / 36, 0, 0, 0, -1. / 4, 0, 0, -1 / 8, -1 / 8],
#          [1 / 19, 4 / 1197, 1 / 252, 0, 0, 1. / 10, 1 / 40, 1 / 10, 1 / 40, -1 / 18, -1. / 36, 0, 0, 0, 1 / 4, 0, 0, 1 / 8, -1 / 8],
#          [1 / 19, 4. / 1197, 1 / 252, 0, 0, 1 / 10, 1. / 40, -1 / 10, -1 / 40, -1 / 18, -1 / 36, 0, 0, 0, -1 / 4, 0, 0, 1 / 8, 1 / 8],
#          [1 / 19, 4 / 1197, 1. / 252, 0, 0, -1 / 10, -1 / 40, -1. / 10, -1 / 40, -1 / 18, -1 / 36, 0, 0, 0, 1 / 4, 0, 0, -1 / 8, 1. / 8]]
#     )
#     names = ['rho', 'e', 'eps', 'jx', 'qx', 'jy', 'qy', 'jz', 'qz', '3pxx', '3pixx', 'pww', 'piww', 'pxy', 'pxz', 'pxx', 'mx', 'my', 'mz']
#     def __init__(self, lattice):
#         assert lattice.stencil == D3Q19
#         super(D3Q19DHumieres, self).__init__(
#             lattice,
#             lattice.convert_to_tensor(self.matrix),
#             lattice.convert_to_tensor(self.inverse)
#         )
#


class D3Q27CentralMomentTransform(Transform):
    def __init__(self,lattice):
        #super(D3Q27CentralMomentTransform, self).__init__(
        #    lattice, None)
        self.lattice = lattice

    class CentralMomentCoefficients:
        def __init__(self,e_minus_u,lattice):
            self.lattice=lattice
            #Precompute the velocity multiplications up to 6th order
            self.cx = e_minus_u[:, 0]
            self.cy = e_minus_u[:, 1]
            self.cz = e_minus_u[:, 2]

            self.cx_cx = self.lattice.einsum('i,i->i', [e_minus_u[:, 0], e_minus_u[:, 0]])
            self.cy_cy = self.lattice.einsum('i,i->i', [e_minus_u[:, 0], e_minus_u[:, 0]])
            self.cz_cz = self.lattice.einsum('i,i->i', [e_minus_u[:, 0], e_minus_u[:, 0]])
            self.cx_cy = self.lattice.einsum('i,i->i', [e_minus_u[:, 0], e_minus_u[:, 1]])
            self.cx_cz = self.lattice.einsum('i,i->i', [e_minus_u[:, 0], e_minus_u[:, 2]])
            self.cy_cz = self.lattice.einsum('i,i->i', [e_minus_u[:, 1], e_minus_u[:, 2]])
            #Order 3
            self.cx_cx_cy = self.lattice.einsum('i,i->i', [self.cx_cx, e_minus_u[:, 1]])
            self.cx_cx_cz = self.lattice.einsum('i,i->i', [self.cx_cx, e_minus_u[:, 2]])
            self.cy_cy_cz = self.lattice.einsum('i,i->i', [self.cy_cy, e_minus_u[:, 2]])
            self.cx_cy_cy = self.lattice.einsum('i,i->i', [self.cy_cy, e_minus_u[:, 0]])
            self.cx_cz_cz = self.lattice.einsum('i,i->i', [self.cz_cz, e_minus_u[:, 0]])
            self.cy_cz_cz = self.lattice.einsum('i,i->i', [self.cz_cz, e_minus_u[:, 1]])
            self.cx_cy_cz = self.lattice.einsum('i,i->i', [self.cx_cy, e_minus_u[:, 2]])
            #Order 4
            self.cx_cx_cy_cy = self.lattice.einsum('i,i->i', [self.cx_cx_cy, e_minus_u[:, 1]])
            self.cx_cx_cz_cz = self.lattice.einsum('i,i->i', [self.cx_cx_cz, e_minus_u[:, 2]])
            self.cy_cy_cz_cz = self.lattice.einsum('i,i->i', [self.cy_cy_cz, e_minus_u[:, 2]])
            self.cx_cx_cy_cz = self.lattice.einsum('i,i->i', [self.cx_cx_cy, e_minus_u[:, 2]])
            self.cx_cy_cy_cz = self.lattice.einsum('i,i->i', [self.cx_cy_cy, e_minus_u[:, 2]])
            self.cx_cy_cz_cz = self.lattice.einsum('i,i->i', [self.cx_cy_cz, e_minus_u[:, 2]])
            #Order 5
            self.cx_cx_cy_cy_cz = self.lattice.einsum('i,i->i', [self.cx_cx_cy_cy, e_minus_u[:, 2]])
            self.cx_cx_cy_cz_cz = self.lattice.einsum('i,i->i', [self.cx_cx_cy_cz, e_minus_u[:, 2]])
            self.cx_cy_cy_cz_cz = self.lattice.einsum('i,i->i', [self.cx_cy_cy_cz, e_minus_u[:, 2]])
            #Order 6
            self.cx_cx_cy_cy_cz_cz = self.lattice.einsum('i,i->i', [self.cx_cx_cy_cy_cz, e_minus_u[:, 2]])

    def transform(self,f):
        CM = torch.zeros([3, 3, 3, f.shape[1],f.shape[2],f.shape[3]])
        e_minus_u = self.lattice.einsum('ij,j...->ij...',[self.lattice.e,-self.lattice.u(f)])

        #Calculate the coefficients cx_cx,cx_cy, etc. for the current time step
        coeff = self.CentralMomentCoefficients(e_minus_u,self.lattice)
        # Order 0
        CM[0,0,0] = 0.0
        # Order 1
        CM[1, 0, 0] = self.lattice.einsum('i,i...->...', [coeff.cx, f])
        CM[0, 1, 0] = self.lattice.einsum('i,i...->...', [coeff.cy, f])
        CM[0, 0, 1] = self.lattice.einsum('i,i...->...', [coeff.cz, f])

        # Order 2
        CM[2, 0, 0] = self.lattice.einsum('i,i...->...', [coeff.cx_cx, f])
        CM[0, 2, 0] = self.lattice.einsum('i,i...->...', [coeff.cy_cy, f])
        CM[0, 0, 2] = self.lattice.einsum('i,i...->...', [coeff.cz_cz, f])
        CM[1, 1, 0] = self.lattice.einsum('i,i...->...', [coeff.cx_cy, f])
        CM[1, 0, 1] = self.lattice.einsum('i,i...->...', [coeff.cx_cz, f])
        CM[0, 1, 1] = self.lattice.einsum('i,i...->...', [coeff.cy_cz, f])

        # Order 3
        CM[2, 1, 0] = self.lattice.einsum('i,i...->...', [coeff.cx_cx_cy, f])
        CM[2, 0, 1] = self.lattice.einsum('i,i...->...', [coeff.cx_cx_cz, f])
        CM[0, 2, 1] = self.lattice.einsum('i,i...->...', [coeff.cy_cy_cz, f])
        CM[1, 2, 0] = self.lattice.einsum('i,i...->...', [coeff.cx_cy_cy, f])
        CM[1, 0, 2] = self.lattice.einsum('i,i...->...', [coeff.cx_cz_cz, f])
        CM[0, 1, 2] = self.lattice.einsum('i,i...->...', [coeff.cy_cz_cz, f])
        CM[1, 1, 1] = self.lattice.einsum('i,i...->...', [coeff.cx_cy_cz, f])

        # Order 4
        CM[2, 2, 0] = self.lattice.einsum('i,i...->...', [coeff.cx_cx_cy_cy, f])
        CM[2, 0, 2] = self.lattice.einsum('i,i...->...', [coeff.cx_cx_cz_cz, f])
        CM[0, 2, 2] = self.lattice.einsum('i,i...->...', [coeff.cy_cy_cz_cz, f])
        CM[2, 1, 1] = self.lattice.einsum('i,i...->...', [coeff.cx_cx_cy_cz, f])
        CM[1, 2, 1] = self.lattice.einsum('i,i...->...', [coeff.cx_cy_cy_cz, f])
        CM[1, 1, 2] = self.lattice.einsum('i,i...->...', [coeff.cx_cy_cz_cz, f])

        # Order 5
        CM[2, 2, 1] = self.lattice.einsum('i,i...->...', [coeff.cx_cx_cy_cy_cz, f])
        CM[2, 1, 2] = self.lattice.einsum('i,i...->...', [coeff.cx_cx_cy_cz_cz, f])
        CM[1, 2, 2] = self.lattice.einsum('i,i...->...', [coeff.cx_cy_cy_cz_cz, f])

        # Order 6
        CM[2, 2, 2] = self.lattice.einsum('i,i...->...', [coeff.cx_cx_cy_cy_cz_cz, f])

        CM /= self.lattice.rho(f)
        return CM

class D3Q27CumulantTransform(Transform):
    def __init__(self,lattice):
        self.lattice = lattice
    def transform(self, f):
        central_moments = D3Q27CentralMomentTransform(self.lattice)
        CM = central_moments.transform(f)
        K = torch.zeros([3, 3, 3, f.shape[1], f.shape[2], f.shape[3]])

        K[0,0,0] = CM[0,0,0]
        K[1,0,0] = CM[1,0,0]
        K[0,1,0] = CM[0,1,0]
        K[0,0,1] = CM[0,0,1]
        K[2,0,0] = CM[2,0,0]
        K[0,2,0] = CM[0,2,0]
        K[0,0,2] = CM[0,0,2]
        K[1,1,0] = CM[1,1,0]
        K[1,0,1] = CM[1,0,1]
        K[0,1,1] = CM[0,1,1]
        K[2,1,0] = CM[2,1,0]
        K[2,0,1] = CM[2,0,1]
        K[0,2,1] = CM[0,2,1]
        K[1, 2, 0] = CM[1, 2, 0]
        K[1, 0, 2] = CM[1, 0, 2]
        K[0, 1, 2] = CM[0, 1, 2]
        K[1, 1, 1] = CM[1, 1, 1]

        K[2, 2, 0] = CM[2, 2, 0] - CM[2, 0, 0] * CM[0, 2, 0] - 2. * CM[1, 1, 0] * CM[1, 1, 0]
        K[2, 0, 2] = CM[2, 0, 2] - CM[2, 0, 0] * CM[0, 0, 2] - 2. * CM[1, 0, 1] * CM[1, 0, 1]
        K[0, 2, 2] = CM[0, 2, 2] - CM[0, 2, 0] * CM[0, 0, 2] - 2. * CM[0, 1, 1] * CM[0, 1, 1]
        K[2, 1, 1] = CM[2, 1, 1] - CM[2, 0, 0] * CM[0, 1, 1] - 2. * CM[1, 1, 0] * CM[1, 0, 1]
        K[1, 2, 1] = CM[1, 2, 1] - CM[0, 2, 0] * CM[1, 0, 1] - 2. * CM[1, 1, 0] * CM[0, 1, 1]
        K[1, 1, 2] = CM[1, 1, 2] - CM[0, 0, 2] * CM[1, 1, 0] - 2. * CM[1, 0, 1] * CM[0, 1, 1]

        K[2, 2, 1] = CM[2, 2, 1] - CM[2, 0, 1] * CM[0, 2, 0] - CM[0, 2, 1] * CM[2, 0, 0] - 2. * CM[2, 1, 0] * CM[
            0, 1, 1] - 2. * CM[1, 2, 0] * CM[1, 0, 1] - 4. * CM[1, 1, 1] * CM[1, 1, 0]
        K[2, 1, 2] = CM[2, 1, 2] - CM[2, 1, 0] * CM[0, 0, 2] - CM[0, 1, 2] * CM[2, 0, 0] - 2. * CM[2, 0, 1] * CM[
            0, 1, 1] - 2. * CM[1, 0, 2] * CM[1, 1, 0] - 4. * CM[1, 1, 1] * CM[1, 0, 1]
        K[1, 2, 2] = CM[1, 2, 2] - CM[1, 2, 0] * CM[0, 0, 2] - CM[1, 0, 2] * CM[0, 2, 0] - 2. * CM[0, 1, 2] * CM[
            1, 1, 0] - 2. * CM[0, 2, 1] * CM[1, 0, 1] - 4. * CM[1, 1, 1] * CM[0, 1, 1]

        K[2, 2, 2] = CM[2, 2, 2] - CM[2, 2, 0] * CM[0, 0, 2] - CM[2, 0, 2] * CM[0, 2, 0] - CM[0, 2, 2] * CM[
            2, 0, 0] - 4. * (CM[2, 1, 1] * CM[0, 1, 1] + CM[1, 2, 1] * CM[1, 0, 1] + CM[1, 1, 2] * CM[1, 1, 0]) - 2. * (
                                 CM[2, 1, 0] * CM[0, 1, 2] + CM[2, 0, 1] * CM[0, 2, 1] + CM[1, 2, 0] * CM[
                             1, 0, 2]) - 4. * CM[1, 1, 1] * \
                     CM[1, 1, 1] + 4. * (
                             CM[2, 0, 0] * CM[0, 1, 1] * CM[0, 1, 1] + CM[0, 2, 0] * CM[1, 0, 1] * CM[1, 0, 1] + CM[
                         0, 0, 2] *
                             CM[1, 1, 0] * CM[1, 1, 0]) + 16. * CM[1, 1, 0] * CM[1, 0, 1] * CM[0, 1, 1] + 2. * CM[
                         2, 0, 0] * CM[0, 2, 0] * CM[0, 0, 2]

        return K

    def equilibrium(self, K):
        Keq = torch.zeros_like(K)
        cs2 = self.lattice.cs * self.lattice.cs
        Keq[0, 0, 0] = 1.0
        Keq[2, 0, 0] = cs2
        Keq[0, 2, 0] = cs2
        Keq[0, 0, 2] = cs2
        return Keq
