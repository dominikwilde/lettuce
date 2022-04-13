import lettuce as lt
import torch
import numpy as np
import matplotlib.pyplot as plt

print("start")

class Oscillation:
    def __init__(self, flow, frequency=1):
        self.frequency = frequency
        self.frequency_lu = frequency/flow.units.convert_time_to_lu(1)
        magnitude = 2 * np.pi * self.frequency
        self.magnitude = flow.units.convert_acceleration_to_lu(magnitude)

# ---------- Set up simulation -------------
device = torch.device("cuda:0")  # replace with device("cpu"), if no GPU is available
lattice = lt.Lattice(lt.D3Q19, device=device, dtype=torch.float32)  # single precision - float64 for double precision
g_lattice = lt.Lattice(lt.D3Q7, device=device, dtype=torch.float32)  # single precision - float64 for double precision
#g_lattice.equilibrium = lt.LinearEquilibrium(g_lattice)

resolution = 400  # resolution of the lattice, low resolution leads to unstable speeds somewhen after 10 (PU)
flow = lt.GravityWaves3D(resolution, 100000, 0.1, lattice)

oscillation = Oscillation(flow=flow, frequency=100)
print("Osci", oscillation.frequency_lu)
do_kbc = False
print("KBC Collision is", do_kbc)

# select collision model - try also KBCCollision or RegularizedCollision
force = lt.ShanChen(lattice,flow.units.relaxation_parameter_lu,[0,0,0])

Fr = 1 #1/oscillation.frequency
print("Froude:", Fr)
drho_dz = -1
print(flow.units.characteristic_length_lu,flow.units.characteristic_velocity_lu,flow.units.convert_time_to_pu(1))
G = -(flow.units.characteristic_velocity_lu/(Fr*flow.units.characteristic_length_lu))**2/drho_dz*flow.units.characteristic_length_lu
g_tau = flow.units.viscosity_lu / g_lattice.stencil.cs ** 2 / 0.7 + 0.5
collision = lt.BGKCollision(lattice, flow.units.relaxation_parameter_lu,g_lattice=g_lattice, force=force, factor=G, g_factor=drho_dz/flow.units.characteristic_length_lu, g_tau=g_tau, do_kbc=do_kbc,oscillation=oscillation)
print(flow.units.characteristic_length_lu/flow.units.convert_time_to_lu(1)**2)
#G = (flow.units.characteristic_velocity_lu/(Fr*flow.units.characteristic_length_lu))**2
#collision = lt.BGKCollision(lattice, flow.units.relaxation_parameter_lu, force=force, gforce=gforce,factor=G, g_factor=flow.units.convert_velocity_to_pu(1))
print("fac,tau,gfac,gtau",collision.factor,collision.tau,collision.g_factor,collision.g_tau)
streaming = lt.StandardStreaming(lattice)
g_streaming = lt.StandardStreaming(g_lattice)

simulation = lt.Simulation(flow, lattice, collision, streaming, g_lattice, g_streaming)
simulation.initialize_f_neq()  # initialize non-conserved moments
# reporters will grab the results in between simulation steps (see io.py and simulation.py)
# Output: Column 1: time in LU, Column 2: kinetic energy in PU

# Output: separate VTK-file with ux,uy,uz and p for every time step in ../data
VTKreport = lt.VTKReporter(lattice, flow, interval=1, filename_base='./data/gravity_{}'.format(Fr))
simulation.reporters.append(VTKreport)
print(collision.factor)
# ---------- Simulate until time = 10 (PU) -------------
print("Simulating", int(simulation.flow.units.convert_time_to_lu(0.3)), "steps! Maybe drink some water in the meantime.")
# runs simulation, but also returns overall performance in MLUPS (million lattice units per second)
print("MLUPS: ", simulation.step(int(simulation.flow.units.convert_time_to_lu(0.3))))

