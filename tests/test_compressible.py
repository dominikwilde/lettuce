from lettuce import compressible, StandardStreaming, BGKCollision, VTKReporter, D2Q9
import matplotlib.pyplot as plt
import numpy as np

import torch
lattice = compressible.CompressibleLattice(compressible.D2Q37,'cpu')

eq = compressible.HermiteEquilibrium(lattice)
a=(eq(1.0,torch.Tensor(np.array([-0.0,1.7])),1.0))
print(a)
print(sum(a))
print(lattice.u(a))
print(lattice.temperature(a))
#coll = BGKCollision(lattice=lattice,tau=0.6)
#streaming = StandardStreaming(lattice)
#flow = compressible.SodShockTube(20,lattice,2)
#sim = compressible.CompressibleSimulation(flow,lattice,coll,streaming)
#rep = VTKReporter(lattice,flow,50,'/scratch/dwilde3m/tmptests/sod/sod')
#sim.reporters.append(rep)
#sim.step(20)
#rho=sim.lattice.rho(sim.f)
#print(rho[0,:,0])