import lettuce as lt
import torch
import numpy as np
import matplotlib.pyplot as plt

print("start")




# ---------- Set up simulation -------------
device = torch.device("cuda:0")  # replace with device("cpu"), if no GPU is available
lattice = lt.Lattice(lt.D3Q27, device=device, dtype=torch.float32)  # single precision - float64 for double precision#
g_lattice = lt.Lattice(lt.D3Q7, device=device, dtype=torch.float32)  # single precision - float64 for double precision
#g_lattice.equilibrium = lt.LinearEquilibrium(g_lattice)

resolution = 128  # resolution of the lattice, low resolution leads to unstable speeds somewhen after 10 (PU)
flow = lt.TaylorGreenVortex3D(resolution, 1600, 0.05, lattice)

do_kbc = False
print("KBC Collision is", do_kbc)

# select collision model - try also KBCCollision or RegularizedCollision
force = lt.ShanChen(lattice,flow.units.relaxation_parameter_lu,[0,0,0])
if do_kbc:
    force = lt.Kupertosh(lattice,flow.units.relaxation_parameter_lu,[0,0,0])


Fr = 0.16
drho_dz=-1
print(flow.units.characteristic_length_lu,flow.units.characteristic_velocity_lu,flow.units.convert_time_to_pu(1))
G = -(flow.units.characteristic_velocity_lu/(Fr*flow.units.characteristic_length_lu))**2/drho_dz*flow.units.characteristic_length_lu
g_tau = flow.units.viscosity_lu / g_lattice.stencil.cs ** 2 / 0.7 + 0.5
collision = lt.BGKCollision(lattice, flow.units.relaxation_parameter_lu,g_lattice=g_lattice, force=force, factor=G, g_factor=drho_dz/flow.units.characteristic_length_lu, g_tau=g_tau, do_kbc=do_kbc)
print(flow.units.characteristic_length_lu/flow.units.convert_time_to_lu(1)**2)
#G = (flow.units.characteristic_velocity_lu/(Fr*flow.units.characteristic_length_lu))**2
#collision = lt.BGKCollision(lattice, flow.units.relaxation_parameter_lu, force=force, gforce=gforce,factor=G, g_factor=flow.units.convert_velocity_to_pu(1))
DensityEnstrophy=lt.DensityEnstrophy(lattice,flow,g_lattice)
deReporter = lt.ObservableReporter(DensityEnstrophy,interval=10, out=None)
print("fac,tau,gfac,gtau",collision.factor,collision.tau,collision.g_factor,collision.g_tau)
streaming = lt.StandardStreaming(lattice)
g_streaming = lt.StandardStreaming(g_lattice)

simulation = lt.Simulation(flow, lattice, collision, streaming, g_lattice, g_streaming)
simulation.initialize_f_neq()  # initialize non-conserved moments
# reporters will grab the results in between simulation steps (see io.py and simulation.py)
# Output: Column 1: time in LU, Column 2: kinetic energy in PU
kinE_reporter = lt.EnergyReporter(lattice, flow, interval=10, out=None)
simulation.reporters.append(kinE_reporter)
enstrophy = lt.Enstrophy(lattice,flow,g_lattice)
potential = lt.PotentialEnergy(lattice,flow,g_lattice)
enstrophyReporter = lt.ObservableReporter(enstrophy,interval=10, out=None)
potentialReporter = lt.ObservableReporter(potential,interval=10, out=None)
timestep = lt.Observable(lattice,flow)
timestepReporter =lt.ObservableReporter(timestep,interval=100)
simulation.reporters.append(timestepReporter)

spectrum = lt.EnergySpectrum(lattice,flow)
spectrumReporter = lt.ObservableReporter(spectrum,interval=int(simulation.flow.units.convert_time_to_lu(10)), out=None)


simulation.reporters.append(deReporter)
simulation.reporters.append(enstrophyReporter)
simulation.reporters.append(potentialReporter)
simulation.reporters.append(spectrumReporter)
# Output: separate VTK-file with ux,uy,uz and p for every time step in ../data
VTKreport = lt.VTKReporter(lattice, flow, interval=int(simulation.flow.units.convert_time_to_lu(5)), filename_base='./data/tgv_{}'.format(Fr))
simulation.reporters.append(VTKreport)
print(collision.factor)
# ---------- Simulate until time = 10 (PU) -------------
print("Simulating", int(simulation.flow.units.convert_time_to_lu(30)), "steps! Maybe drink some water in the meantime.")
# runs simulation, but also returns overall performance in MLUPS (million lattice units per second)
print("MLUPS: ", simulation.step(int(simulation.flow.units.convert_time_to_lu(30))))

# ---------- Plot kinetic energy over time (PU) -------------
# grab output of kinetic energy reporter
E = np.asarray(kinE_reporter.out)
# save kinetic energy values for later use
namestring = "TGV_Q" + str(lattice.Q) + "x"+ str(g_lattice.Q) + "_N" + str(resolution) + "_KBC" + str(do_kbc) + "_" + collision.force.__class__.__name__ + "_Fr" + str(Fr)

np.save(namestring+"_KinEnergy", E)

E = np.asarray(deReporter.out)
# save kinetic energy values for later use
np.save(namestring+"_DensityGradient", E)

E = np.asarray(enstrophyReporter.out)
# normalize to size of grid, not always necessary
np.save(namestring+"_Enstrophy", E)

E = np.asarray(potentialReporter.out)
# normalize to size of grid, not always necessary
np.save(namestring+"_PotentialEnergy", E)

E = np.asarray(spectrumReporter.out)
# normalize to size of grid, not always necessary
np.save(namestring+"_Spectrum", E)
