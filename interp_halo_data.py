import scipy.io
import numpy as np
import numpy.linalg as la
from sympy import *
import os
import os.path
from scipy.linalg import lu_factor, lu_solve
from STMint.STMint import STMint
import matplotlib.pyplot as plt
import time



#class containing the variational data and ability to solve BVPs
class OrbitVariationalData:
	#lowest level STMs, exponent for 2^exponent of these STMs
	def __init__(self, STTs, STMs, trvs, T, exponent):
		self.STMs = STMs
		#stts are energy output only
		self.STTs = STTs
		self.T = T
		self.ts = trvs[:,0]
		self.rs = trvs[:,1:4]
		self.vs = trvs[:,4:]
		self.exponent = exponent
		self.refinedList = [STMs]
		self.refinedListSTTs = [STTs]
		self.constructAdditionalLevels()
	def cocycle2(self, stm10, stt10, stm21, stt21):
		""" Function to find STM and STT along two combined subintervals

	   The cocycle conditon equation is used to find Phi(t2,t0)=Phi(t2,t1)*Phi(t1,t0)
		and the generalized cocycle condition is used to find Psi(t2,t0)

		Args:
			stm10 (np array)
				State transition matrix from time 0 to 1

			stt10 (np array)
				State transition tensor from time 0  to 1

			stm21 (np array)
				State transition matrix from time 1 to 2

			stt21 (np array)
				State transition tensor from time 1 to 2

		Returns:
			stm20 (np array)
				State transition matrix from time 0 to 2
			stt20 (np array)
				State transition tensor from time 0 to 2            
		"""
		stm20 = np.matmul(stm21, stm10)
		#stt20 = np.einsum('il,ljk->ijk', stm21, stt10) + np.einsum('ilm,lj,mk->ijk', stt21, stm10, stm10)
		#cocycles for stt with energy output only
		stt20 = stt10 + np.einsum('lm,lj,mk->jk', stt21, stm10, stm10)
		return [stm20, stt20]
	#create structure with most refined STMs at [0], and the monodromy matrix at [exponent] 
	def constructAdditionalLevels(self):
		for i in range(self.exponent):
			stms1 = []
			stts1 = []
			for j in range(2**(self.exponent-i-1)):
				STMCombined, STTCombined = self.cocycle2(self.refinedList[i][2*j], self.refinedListSTTs[i][2*j], self.refinedList[i][2*j+1], self.refinedListSTTs[i][2*j+1])
				stms1.append(STMCombined)
				stts1.append(STTCombined)
			self.refinedListSTTs.append(stts1)
			self.refinedList.append(stms1)
	#takes self.exponent number matrix mults/cocycle conditions in a binary search style
	def findSTMAux(self, t0, tf):
		foundCoarsestLevel = False
		for i in range(self.exponent+1):
			j = self.exponent - i
			stepLength = self.T / (2.**i)
			location0 = (int) (t0 // stepLength)
			locationf = (int) (tf // stepLength)
			#this is the coarsest level at which there is a full precomputed STM between the two times
			if not foundCoarsestLevel:
				if locationf-location0 >= 2:
					foundCoarsestLevel = True
					leftPoint = (location0+1)*(2**j)
					rightPoint = locationf*(2**j)
					stm = self.refinedList[j][location0+1]
					stt = self.refinedListSTTs[j][location0+1]
					# if more than 2 periods
					if locationf-location0 == 3:
						stm, stt = self.cocycle2(stm, stt, self.refinedList[j][location0+2], self.refinedListSTTs[j][location0+2])
			else:
				#left and right points of the already constructed STM
				lp = (int) (leftPoint // ((2**j)))
				rp = (int) (rightPoint // ((2**j)))
				if lp-location0 == 2:
					stm, stt = self.cocycle2(self.refinedList[j][location0+1], self.refinedListSTTs[j][location0+1], stm, stt)
					leftPoint -= (2**j)
				if locationf-rp == 1:
					stm, stt = self.cocycle2(stm, stt, self.refinedList[j][rp], self.refinedListSTTs[j][rp])
					rightPoint += (2**j)
		#approximate at the endpoints
		if not foundCoarsestLevel:
			smallestPieceTime = self.T / 2.**self.exponent
			location0 = (int) (t0 // smallestPieceTime)
			locationf = (int) (tf // smallestPieceTime)
			if location0 == locationf:
				stm = np.identity(12) + (tf-t0)/smallestPieceTime * (self.STMs[location0]-np.identity(12))
				stt = (tf-t0)/smallestPieceTime * self.STTs[location0]
			else:
				line = smallestPieceTime*locationf
				stmLeft = np.identity(12) + (line-t0)/smallestPieceTime * (self.STMs[location0]-np.identity(12))
				sttLeft = (line-t0)/smallestPieceTime * self.STTs[location0]
				stmRight = np.identity(12) + (tf-line)/smallestPieceTime * (self.STMs[locationf]-np.identity(12))
				sttRight = (tf-line)/smallestPieceTime * self.STTs[locationf]
				stm, stt = self.cocycle2(stmLeft, sttLeft, stmRight, sttRight)
		else:
			smallestPieceTime = self.T / 2.**self.exponent
			leftPointT = smallestPieceTime * leftPoint
			rightPointT = smallestPieceTime * rightPoint
			leftContribution = np.identity(12) + (leftPointT-t0)/smallestPieceTime * (self.STMs[leftPoint-1]-np.identity(12))
			leftContributionSTT = (leftPointT-t0)/smallestPieceTime * (self.STTs[leftPoint-1])
			stm, stt = self.cocycle2(leftContribution, leftContributionSTT, stm, stt)
			rightContribution = np.identity(12) + (tf-rightPointT)/smallestPieceTime * (self.STMs[rightPoint]-np.identity(12))
			rightContributionSTT = (tf-rightPointT)/smallestPieceTime * self.STTs[rightPoint]
			stm, stt = self.cocycle2(stm, stt, rightContribution, rightContributionSTT)
		return stm, stt
	#calculate the STM (and STT) for a given start and end time
	def findSTM(self, t0, tf):
		if t0 > tf:
			print("STM requested with t0>tf.")
		left = (int) (t0 // self.T)
		right = (int) (tf // self.T)
		t0 = t0 % self.T 
		tf = tf % self.T
		if left == right:
			stm, stt = self.findSTMAux(t0, tf)
		else:
			stm, stt = self.findSTMAux(t0, self.T-10E-12)
			stmf, sttf = self.findSTMAux(0., tf)
			if right-left > 1:
				#stmmid = np.linalg.matrix_power(self.refinedList[-1][0], right-left-1)
				stmmid = self.refinedList[-1][0]
				sttmid = self.refinedListSTTs[-1][0]
				for i in range(right-left-1):
					stmmid, sttmid = self.cocycle2(stmmid, sttmid, self.refinedList[-1][0], self.refinedListSTTs[-1][0])
				stm, stt = self.cocycle2(stm, stt, stmmid, sttmid)
			stm, stt = self.cocycle2(stm, stt, stmf, sttf)
		return stm, stt

	#find relative rotating frame velocity that gives inertial relative velocity of zero
	def findRotRelVel(self, rrel):
		return np.array([rrel[1], -1.*rrel[0], 0.])
	#precompute necessary quantities for repeated calling of different transfers in same time ranges
	def precompute_lu(self, t0, tf):
		stm, stt = self.findSTM(t0, tf)
		lu, piv = lu_factor(stm[:6,6:12])
		return (stm[:6,:6], stm[6:12,:6], stm[6:12,6:], stt, lu, piv)
		
	#find the approximate cost of a relative transfer in mps (for repeated calls with same initial and final times)
	#positions supplied in au
	#rotating frame velocities in canonical units
	#return delta-vs in canonical units
	def solve_bvp_cost(self, stmxx, stmlx, stmll, stt, lu, piv, x0rel, xfrel):
		l0rel = lu_solve((lu, piv), xfrel - np.matmul(stmxx, x0rel))
		relAugState = np.concatenate((x0rel, l0rel))
		return np.einsum("jk,j,k->", stt, relAugState, relAugState)
		
		
	#find the approximate cost of a relative transfer (for repeated calls with same initial and final times)
	#positions supplied in km
	#Assume inertial relative velocities are zero
	#return energy in canonical units
	def solve_bvp_cost_convenience(self, precomputeData, r0rel, rfrel):
		(stmxx, stmlx, stmll, stt, lu, piv) = precomputeData
		r0rel = self.posKMtoAU(r0rel)
		rfrel = self.posKMtoAU(rfrel)
		v0rel = self.findRotRelVel(r0rel)
		vfrel = self.findRotRelVel(rfrel)
		#all in canonical units at this point-can be fed into bvp_solver function
		en = self.solve_bvp_cost(stmxx, stmlx, stmll, stt, lu, piv, np.concatenate((r0rel, v0rel)), np.concatenate((rfrel, vfrel)))
		return en

	#convert delta-v cost to m/s
	def veltoMpS(self, vel):
		return vel*29784.
	#convert position from KM to AU
	def posKMtoAU(self, pos):
		return pos / 149597870.7

		
def optControlDynamics():
	x,y,z,vx,vy,vz,lx,ly,lz,lvx,lvy,lvz,En=symbols("x,y,z,vx,vy,vz,lx,ly,lz,lvx,lvy,lvz,En")
	mu = 3.00348e-6
	mu1 = 1. - mu
	mu2 = mu
	r1 = sqrt((x + mu2)**2 + (y**2) + (z**2))
	r2 = sqrt((x - mu1)**2 + (y**2) + (z**2))
	U = (-1/2)*(mu1*(r1**2) + mu2*(r2**2)) - (mu1/r1) - (mu2/r2)
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
#initial conditions for sun earth halo orbit
ics = [1.00822114953991, 0., -0.001200000000000000, 0., 0.010290010931740649, 0., 0., 0., 0., 0., 0., 0., 0.]
T = 3.1002569555488506
#ics = [1.0079262817004409, 0., -0.002000000000000000, 0., 0.011250447954520019, 0., 0., 0., 0., 0., 0., 0., 0.]
#T = 3.095762839166285
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
		states.append(state[0])
		STMs.append(STM[:12,:12])
		STTs.append(STT[12,:12,:12])
		curState = state[0]
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
print("energy in cm^2/s^3")
print(en*10000.*177.004)


#plot transfer costs as a function of transfer time (between 1 and 10 weeks)
ts = np.arange(t0+7.*2.*np.pi/365., 10.*7.*2.*np.pi/365., 0.01)
ens = list(map(lambda t:orb.solve_bvp_cost_convenience(orb.precompute_lu(t0, t), r0rel, rfrel), ts))
plt.figure()
plt.plot(ts, ens)
plt.show()
