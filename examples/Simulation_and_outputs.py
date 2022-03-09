import lettuce as lt
import torch
import numpy as np
import matplotlib.pyplot as plt

print("start")

# ---------- Set up simulation -------------
device = torch.device("cuda:0")  # replace with device("cpu"), if no GPU is available
lattice = lt.Lattice(lt.D3Q27, device=device, dtype=torch.float32)  # single precision - float64 for double precision
resolution = 128  # resolution of the lattice, low resolution leads to unstable speeds somewhen after 10 (PU)
flow = lt.TaylorGreenVortex3D(resolution, 1600, 0.1, lattice)

# select collision model - try also KBCCollision or RegularizedCollision
force = lt.ShanChen(lattice,flow.units.relaxation_parameter_lu,[0,0,0])
gforce = lt.ShanChen(lattice,flow.units.relaxation_parameter_lu,[0,0,0])

G = 1.001
collision = lt.BGKCollision(lattice, flow.units.relaxation_parameter_lu, force=force, gforce=gforce,factor=flow.units.convert_acceleration_to_lu(G), g_factor=flow.units.convert_acceleration_to_lu(G)/flow.units.characteristic_velocity_lu)
print("fac,tau,gfac,gtau",collision.factor,collision.tau,collision.g_factor,collision.g_tau)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow, lattice, collision, streaming)
simulation.initialize_f_neq()  # initialize non-conserved moments

# reporters will grab the results in between simulation steps (see io.py and simulation.py)
# Output: Column 1: time in LU, Column 2: kinetic energy in PU
kinE_reporter = lt.EnergyReporter(lattice, flow, interval=1, out=None)
simulation.reporters.append(kinE_reporter)
# Output: separate VTK-file with ux,uy,uz and p for every time step in ../data
VTKreport = lt.VTKReporter(lattice, flow, interval=100, filename_base='./data/output_{}'.format(G))
simulation.reporters.append(VTKreport)
print(collision.factor)
# ---------- Simulate until time = 10 (PU) -------------
print("Simulating", int(simulation.flow.units.convert_time_to_lu(20)), "steps! Maybe drink some water in the meantime.")
# runs simulation, but also returns overall performance in MLUPS (million lattice units per second)
print("MLUPS: ", simulation.step(int(simulation.flow.units.convert_time_to_lu(20))))

# ---------- Plot kinetic energy over time (PU) -------------
# grab output of kinetic energy reporter
E = np.asarray(kinE_reporter.out)
# normalize to size of grid, not always necessary
E[:, 1] = E[:, 1] / (2 * np.pi) ** 3
# save kinetic energy values for later use
np.save("TGV3DoutRes" + str(resolution) + "G" + str(G), E)
