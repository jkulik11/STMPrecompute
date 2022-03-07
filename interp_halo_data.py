import scipy.io
import numpy as np
import numpy.linalg as la
import os
import os.path
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt


#class containing the variational data and ability to solve BVPs
class OrbitVariationalData:
	#lowest level STMs, exponent for 2^exponent of these STMs
	def __init__(self, STMs, trvs, T, exponent):
		self.STMs = STMs
		self.T = T
		self.ts = trvs[:,0]
		self.rs = trvs[:,1:4]
		self.vs = trvs[:,4:]
		self.exponent = exponent
		self.refinedList = [STMs]
		self.constructAdditionalLevels()
	#create structure with most refined STMs at [0], and the monodromy matrix at [exponent] 
	def constructAdditionalLevels(self):
		for i in range(self.exponent):
			stms1 = []
			for j in range(2**(self.exponent-i-1)):
				stms1.append(np.matmul(self.refinedList[i][2*j+1], self.refinedList[i][2*j]))
			self.refinedList.append(stms1)
	#takes self.exponent number matrix mults in a binary search style
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
					# if more than 2 periods
					if locationf-location0 == 3:
						stm = np.matmul(self.refinedList[j][location0+2], stm)
						#stm = la.matrix_power(stm, locationf-location0-1)
			else:
				#left and right points of the already constructed STM
				lp = (int) (leftPoint // ((2**j)))
				rp = (int) (rightPoint // ((2**j)))
				if lp-location0 == 2:
					stm = np.matmul(stm, self.refinedList[j][location0+1])
					leftPoint -= (2**j)
				if locationf-rp == 1:
					stm = np.matmul(self.refinedList[j][rp], stm)
					rightPoint += (2**j)
		#approximate at the endpoints
		if not foundCoarsestLevel:
			smallestPieceTime = self.T / 2.**self.exponent
			location0 = (int) (t0 // smallestPieceTime)
			locationf = (int) (tf // smallestPieceTime)
			if location0 == locationf:
				stm = np.identity(6) + (tf-t0)/smallestPieceTime * (self.STMs[location0]-np.identity(6))
			else:
				line = smallestPieceTime*locationf
				stmLeft = np.identity(6) + (line-t0)/smallestPieceTime * (self.STMs[location0]-np.identity(6))
				stmRight = np.identity(6) + (tf-line)/smallestPieceTime * (self.STMs[locationf]-np.identity(6))
				stm = np.matmul(stmRight, stmLeft)
		else:
			smallestPieceTime = self.T / 2.**self.exponent
			leftPointT = smallestPieceTime * leftPoint
			rightPointT = smallestPieceTime * rightPoint
			leftContribution = np.identity(6) + (leftPointT-t0)/smallestPieceTime * (self.STMs[leftPoint-1]-np.identity(6))
			stm = np.matmul(stm, leftContribution)
			rightContribution = np.identity(6) + (tf-rightPointT)/smallestPieceTime * (self.STMs[rightPoint]-np.identity(6))
			stm = np.matmul(rightContribution, stm)
		return stm
	#calculate the STM for a given start and end time
	def findSTM(self, t0, tf):
		if t0 > tf:
			print("STM requested with t0>tf.")
		left = (int) (t0 // self.T)
		right = (int) (tf // self.T)
		t0 = t0 % self.T 
		tf = tf % self.T
		if left == right:
			stm = self.findSTMAux(t0, tf)
		else:
			stm = self.findSTMAux(t0, self.T-10E-12)
			stmf = self.findSTMAux(0., tf)
			if right-left > 1:
				stmmid = np.linalg.matrix_power(self.refinedList[-1][0], right-left-1)
				stm = np.matmul(stmmid, stm)
			stm = np.matmul(stmf, stm)
		return stm
		
	#TODO: switch to astropy unit conversions
	#convert delta-v cost to m/s
	def veltoMpS(self, vel):
		return vel*29784.
	#convert position from KM to AU
	def posKMtoAU(self, pos):
		return pos / 149597870.7

	#find relative rotating frame velocity that gives inertial relative velocity of zero
	def findRotRelVel(self, rrel):
		return np.array([rrel[1], -1.*rrel[0], 0.])
	#precompute necessary quantities for repeated calling of different transfers in same time ranges
	def precompute_lu(self, t0, tf):
		stm = self.findSTM(t0, tf)
		lu, piv = lu_factor(stm[:3,3:])
		return (stm[:3,:3], stm[3:,:3], stm[3:,3:], lu, piv)
		
	#find the approximate cost of a relative transfer in mps (for repeated calls with same initial and final times)
	#positions supplied in au
	#rotating frame velocities in canonical units
	#return delta-vs in canonical units
	def solve_bvp_cost(self, stmrr, stmvr, stmvv, lu, piv, r0rel, v0rel, rfrel, vfrel):
		v0relNew = lu_solve((lu, piv), rfrel - np.matmul(stmrr, r0rel))
		vfrelNew = np.matmul(stmvr, r0rel) + np.matmul(stmvv, v0relNew)
		return (v0relNew-v0rel, vfrelNew-vfrel)
		
	#find the approximate cost of a relative transfer in mps (for repeated calls with same initial and final times)
	#positions supplied in km
	#Assume inertial relative velocities are zero
	#return delta-v in m/s
	def solve_bvp_cost_convenience(self, precomputeData, r0rel, rfrel):
		(stmrr, stmvr, stmvv, lu, piv) = precomputeData
		r0rel = self.posKMtoAU(r0rel)
		rfrel = self.posKMtoAU(rfrel)
		v0rel = self.findRotRelVel(r0rel)
		vfrel = self.findRotRelVel(rfrel)
		#all in canonical units at this point-can be fed into bvp_solver function
		dvsCanon = self.solve_bvp_cost(stmrr, stmvr, stmvv, lu, piv, r0rel, v0rel, rfrel, vfrel)
		return (self.veltoMpS(dvsCanon[0]), self.veltoMpS(dvsCanon[1]))
		



#store precomputed data in the following files
fileName = "halo"
trvFileName = "./" + fileName + "_trvs.mat"
STMFileName = "./" + fileName + "_STMs.mat"
#find a SEL2 Halo with .002 amplitude in canonical units
Az=0.002
#use 2^8 subdivisions when calculating the values of the STM
exponent=8

#call wolfram script if data file does not already exist
if not os.path.isfile(trvFileName): 
#1 denotes that the halo orbit is found rather than an ic being given
#2^8 samples along a Northern SEL2 halo with .002 amplitude
	os.system(' '.join(['wolframscript produce_halo_data.wls', '1', 'SE', 'S', str(Az), fileName, str(exponent)]))

#load data from file
trvmat = list(scipy.io.loadmat(trvFileName).values())[0]
#period
T = trvmat[-1, 0]
#Take off last element which is same as first element up to integration error tolerances (periodicity)
trvmat = trvmat[:-1]

#remove identity element at beginning
STMmat = list(scipy.io.loadmat(STMFileName).values())[1:]

#initialize object used for computation
orb = OrbitVariationalData(STMmat, trvmat, T, exponent)


#test on a 100k km transfer in a couple weeks
t0 = 7.*2.*np.pi/365.
#two weeks
tf = 3*7.*2.*np.pi/365.
r0rel = np.array([-100000.,0.,0.])
rfrel = np.array([0,0,100000])
#precompute variational data for this time period
precomputeData = orb.precompute_lu(t0, tf)
#use for this one boundary value solution (same variational data could be used multiple times)
dvs = orb.solve_bvp_cost_convenience(precomputeData, r0rel, rfrel)
print(dvs)


#lets plot transfer costs as a function of transfer time to make sure the result looks reasonable
ts = np.arange(t0, 3.5*T, 0.01)
#print(list(map(lambda t:1+t, ts)))
dvs = list(map(lambda t:orb.solve_bvp_cost_convenience(orb.precompute_lu(0., t), r0rel, rfrel), ts))
dvnorms0 = list(map(lambda l:min(100, la.norm(l[0])), dvs))
dvnormsf = list(map(lambda l:min(100, la.norm(l[1])), dvs))
plt.figure()
plt.plot(ts, dvnorms0)
plt.plot(ts, dvnormsf)
plt.show()


#testing cases

#test = orb.findSTM(0,7.0000*T/2.**exponent)
#check = np.matmul(STMmat[6], np.matmul(orb.refinedList[1][2],orb.refinedList[2][0]))

#test = orb.findSTM(0,T-.000000000001)
#check = orb.refinedList[-1][0]

#test = orb.findSTM(0,T/2.+.000000000001)
#check = orb.refinedList[-2][0]

#test = orb.findSTM(0,3.*T/2.)
#check = np.matmul(orb.refinedList[-2][0], orb.refinedList[-1][0])

#test = orb.findSTM(T/2.,3.*T/2.)
#check = np.matmul(orb.refinedList[-2][0], orb.refinedList[-2][1])

#print(test)
#print(check)
#print(test-check)



