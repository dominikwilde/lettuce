from lettuce import compressible, StandardStreaming, BGKCollision, VTKReporter, D2Q9
import matplotlib.pyplot as plt
import numpy as np

import torch
lattice = compressible.CompressibleLattice(compressible.D2Q37,'cpu')

# eq = compressible.HermiteEquilibrium(lattice)
# a=(eq(1.0,torch.Tensor(np.array([-0.0,1.7])),1.0,4))
# print(a)
# print(sum(a))
# print(lattice.u(a))
# print(lattice.temperature(a))

class BoundaryManipulator:
    """Reports the kinetic energy with respect to analytic solution."""
    def __init__(self, lattice, f, g, mask):
        self.lattice = lattice
        self.f_begin = f
        self.g_begin = g
        self.mask = lattice.convert_to_tensor(mask)

    def __call__(self, i, t, f, g,simulation):
        f = torch.where(self.no_collision_mask, f, self.f_begin)
        g = torch.where(self.no_collision_mask, g, self.g_begin)
        return f,g



coll = compressible.BGKCompressibleCollision(lattice=lattice,tau=0.8, gamma = 1.4)
streaming = StandardStreaming(lattice)
flow = compressible.SodShockTube(200,lattice,T_left=1.25,rho_left=8)
sim = compressible.CompressibleSimulation(flow,lattice,coll,streaming)
#rep = VTKReporter(lattice,flow,1,'/home/dwilde3m/tmp/sod/sod')
#sim.reporters.append(rep)
print(coll.tau/3-0.5/3)
plt.show()
sim.step(30)
rho=sim.lattice.rho(sim.f)
T=sim.lattice.temperature(sim.f,sim.g,2.5)
u=sim.lattice.u(sim.f)
#plt.plot(rho)
plt.interactive(False)
plt.plot((u[0,:,0]).cpu())
print((np.var(u[0,145:175,0].numpy())))
#plt.xlim(0,300)
#plt.plot((rho[0,:,0]).cpu())
plt.show()