from lettuce import compressible, StandardStreaming, BGKCollision, VTKReporter, D2Q9
import matplotlib.pyplot as plt
import numpy as np

import torch
lattice = compressible.CompressibleLattice(compressible.D2Q37,'cuda:0')

# eq = compressible.HermiteEquilibrium(lattice)
# a=(eq(1.0,torch.Tensor(np.array([-0.0,1.7])),1.0,4))
# print(a)
# print(sum(a))
# print(lattice.u(a))
# print(lattice.temperature(a))

coll = compressible.BGKCompressibleCollision(lattice=lattice,tau=0.55, gamma = 1.4)
streaming = StandardStreaming(lattice)
flow = compressible.SodShockTube(200,lattice,T_left=1.25,rho_left=8.0)
sim = compressible.CompressibleSimulation(flow,lattice,coll,streaming)
#rep = VTKReporter(lattice,flow,1,'/home/dwilde3m/tmp/sod/sod')
#sim.reporters.append(rep)
sim.step(20)
rho=sim.lattice.rho(sim.f)
T=sim.lattice.temperature(sim.f,sim.g,2.5)
u=sim.lattice.u(sim.f)
#plt.plot(rho)
plt.interactive(False)
plt.plot((T[0,:,0]).cpu())
#plt.plot((rho[0,:,0]).cpu())
plt.show()