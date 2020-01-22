from lettuce import compressible, StandardStreaming, BGKCollision, VTKReporter

import torch
lattice = compressible.Lattice(compressible.D2Q37,'cuda:0')
coll = BGKCollision(lattice=lattice,tau=0.6)
streaming = StandardStreaming(lattice)
flow = compressible.SodShockTube(500,lattice,2)
sim = compressible.CompressibleSimulation(flow,lattice,coll,streaming)
rep = VTKReporter(lattice,flow,50,'/scratch/dwilde3m/tmptests/sod/sod')
sim.reporters.append(rep)
sim.step(2000)