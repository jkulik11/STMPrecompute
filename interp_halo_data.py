import scipy.io
import numpy as np
import numpy.linalg as la
from sympy import *
import os
import os.path
from scipy.linalg import lu_factor, lu_solve
from STMint.STMint import STMint
import matplotlib.pyplot as plt
from OrbitVariationalData import OrbitVariationalData
		
#symbolically define the dynamics for energy optimal control in the cr3bp
#this will be used by the STMInt package for numerically integrating the STM and STT
def optControlDynamics():
	x,y,z,vx,vy,vz,lx,ly,lz,lvx,lvy,lvz,En=symbols("x,y,z,vx,vy,vz,lx,ly,lz,lvx,lvy,lvz,En")
	mu = 3.00348e-6
	mu1 = 1. - mu
	mu2 = mu
	r1 = sqrt((x + mu2)**2 + (y**2) + (z**2))
	r2 = sqrt((x - mu1)**2 + (y**2) + (z**2))
	U = (-1./2.)*(x**2+y**2) - (mu1/r1) - (mu2/r2)
	dUdx = diff(U,x)
	dUdy = diff(U,y)
	dUdz = diff(U,z)

	RHS = Matrix([vx,vy,vz,((-1*dUdx) + 2*vy),((-1*dUdy)- 2*vx),(-1*dUdz)])

	variables = Matrix([x,y,z,vx,vy,vz,lx,ly,lz,lvx,lvy,lvz,En])

	dynamics = Matrix(BlockMatrix([[RHS - Matrix([0,0,0,lvx,lvy,lvz])], 
		[-1.*RHS.jacobian(Matrix([x,y,z,vx,vy,vz]).transpose())*Matrix([lx,ly,lz,lvx,lvy,lvz])],
		[.5*Matrix([lvx**2+lvy**2+lvz**2])]]))
	#return Matrix([x,y,z,vx,vy,vz]), RHS
	return variables, dynamics


#store precomputed data in the following files
fileName = "haloEnergy"
trvFileName = "./" + fileName + "_trvs.mat"
STMFileName = "./" + fileName + "_STMs.mat"
STTFileName = "./" + fileName + "_STTs.mat"
#initial conditions for a sun-earth halo orbit
#with zero initial conditions for costates and energy
ics = [1.00822114953991, 0., -0.001200000000000000, 0., 0.010290010931740649, 0., 0., 0., 0., 0., 0., 0., 0.]
T = 3.1002569555488506
#use 2^8 subdivisions when calculating the values of the STM
exponent=8



#precompute variational data if data file does not already exist
if not os.path.isfile(trvFileName):
	variables, dynamics = optControlDynamics()
	threeBodyInt = STMint(variables, dynamics, variational_order=2)
	t_step = T/2.**exponent
	curState = ics
	states = [ics]
	#STMs = [np.identity(12)]
	#STTs = [np.zeros((12,12))]
	STMs = []
	STTs = []
	tVals = np.linspace(0, T, num = 2**exponent + 1)
	for i in range(2**exponent):
		[state, STM, STT] = threeBodyInt.dynVar_int2([0,t_step], curState, output='final', max_step=.001)
		states.append(state)
		STMs.append(STM[:12,:12])
		STTs.append(STT[12,:12,:12])
		curState = state
	scipy.io.savemat(trvFileName, {"trvs": np.hstack((np.transpose(np.array([tVals])), states))})
	scipy.io.savemat(STMFileName, {"STMs": STMs})
	scipy.io.savemat(STTFileName, {"STTs": STTs})


#load data from file
trvmat = list(scipy.io.loadmat(trvFileName).values())[-1]
#period
T = trvmat[-1, 0]
#Take off last element which is same as first element up to integration error tolerances (periodicity)
trvmat = trvmat[:-1]

STMmat = list(scipy.io.loadmat(STMFileName).values())[-1]
STTmat = list(scipy.io.loadmat(STTFileName).values())[-1]
#initialize object used for computation
orb = OrbitVariationalData(STTmat, STMmat, trvmat, T, exponent)


#test on a 10k km transfer in two weeks
t0 = 1.*7.*2.*np.pi/365.
#two weeks
tf = 3.*7.*2.*np.pi/365.
r0rel = np.array([10000.,0.,0.])
rfrel = np.array([0,0,0])
#precompute variational data for this time period
precomputeData = orb.precompute_lu(t0, tf)
#use for this one boundary value solution (same variational data could be used multiple times)
en = orb.solve_bvp_cost_convenience(precomputeData, r0rel, rfrel)
#energy in cm^2/s^3
print("Energy in cm^2/s^3")
print(en*10000.*177.004)


#plot transfer costs as a function of transfer time (between 1 and 10 weeks)
ts = np.arange(t0+7.*2.*np.pi/365., 10.*7.*2.*np.pi/365., 0.01)
ens = list(map(lambda t:orb.solve_bvp_cost_convenience(orb.precompute_lu(t0, t), r0rel, rfrel), ts))
plt.figure()
plt.plot(ts, ens)
plt.xlabel('Rendezvous Time (TU)')
plt.ylabel('Energy Cost (DU^2/TU^3)')
plt.title("Rendezvous Cost")

plt.show()
