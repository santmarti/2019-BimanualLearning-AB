import sys
import os.path
import numpy as np
import pickle
import math
from explauto import Environment
from explauto import SensorimotorModel
from explauto import InterestModel

def distFwdNN(sm_model,x):
    dists, indexes = sm_model.model.imodel.fmodel.dataset.nn_x(x, k=1)
    return dists[0]
    #print "Distance in Motor space:", dists[0]

def distInvNN(sm_model,y):
    dists, indexes = sm_model.model.imodel.fmodel.dataset.nn_y(y, k=1)
    return dists[0]

def distFwdNNContext(sm_model,x,c,c_dims):
	dists, index = sm_model.model.imodel.fmodel.dataset.nn_dims(x,c,
		range(sm_model.model.imodel.fmodel.dim_x),
		list(np.array(c_dims) + sm_model.model.imodel.fmodel.dim_x),
		k=1)
	return dists[0]

def errorInvFwd(sm_model,y):
	x = sm_model.inverse_prediction(y)
	
	y_pred = sm_model.forward_prediction(x) # environment call

	return np.linalg.norm(y-y_pred)

def dist(p,q):
    return math.hypot(q[0] - p[0], q[1] - p[1])

def min_max(v,mins,maxs):
    v = np.minimum(v, maxs)
    v= np.maximum(v, mins)
    return v

def printProgress(i=0, val=None, valp=None):
    if(i%10==0): 
        print ".",
        if(i%50==0): 
            print i,
            if(i>0 and val!=None and valp==None): print "(",val,")",
            elif(i>0 and val!=None and valp!=None): print "(",val,",",valp,")",
        sys.stdout.flush()


class ErrorManager:

	def iniHist(self):
		self.error_hist = {}
		self.evaluation_hist = {}
		for i in range(len(self.models)): 
			self.error_hist[i]=[]
			self.evaluation_hist[i]=[]

	def __init__(self,inclass,folder='human'):
		self.folder = folder
		self.models = inclass.models
		self.im_goals = {}
		for i in range(len(self.models)): self.im_goals[i]=[]
		self.im_models = inclass.im_models
		self.env = inclass.env
		self.in_mins = inclass.in_mins
		self.in_maxs = inclass.in_maxs
		self.out_mins = inclass.out_mins
		self.out_maxs = inclass.out_maxs
		self.strid = inclass.strid
		self.min_err = 0.01
		self.iniHist()

	def setExp(self,folder,strid):
		self.folder = folder
		self.strid = strid

	def load_errors(self,imodel):
		strid = self.strid
		fname = "data/%s/error/%s-model%d-error.data"%(self.folder,self.strid,imodel)
		return pickle.load(open(fname, "rb"))

	def save_errors(self,imodel):
		folder = self.folder
		strid = self.strid
		errlist = self.error_hist[imodel]

		errordir = "data/%s/error"%folder
		if not os.path.exists(errordir): os.makedirs(errordir)

		fname = "%s/%s-model%d-error.data"%(errordir,strid,imodel)
		pickle.dump(errlist, open(fname, "wb"))

	def load_eval(self,imodel):
		strid = self.strid
		fname = "data/%s/error/%s-model%d-eval.data"%(self.folder,strid,imodel)
		return pickle.load(open(fname, "rb"))

	def save_eval(self,imodel):
		strid = self.strid
		evalist = self.evaluation_hist[imodel]
		if(len(evalist)<1): return
		fname = "data/%s/error/%s-model%d-eval.data"%(self.folder,strid,imodel)
		pickle.dump(evalist, open(fname, "wb"))

	def save_goals(self,imodel):
		strid = self.strid
		goals = self.im_goals[imodel]
		if(len(goals)<1): return
		fname = "data/%s/error/%s-model%d-goals.data"%(self.folder,strid,imodel)
		pickle.dump(goals, open(fname, "wb"))


	def errLog(self,err,imodel):
		errlist = self.error_hist[imodel]
		if(len(errlist) > 20): var = np.var(errlist[-20:])
		elif(len(errlist) > 1): var = np.var(errlist)
		else: var = 0

		errlist.append([err,var])
		if(len(errlist)%20==0): 
			self.save_errors(imodel)
			self.save_eval(imodel)
			self.save_goals(imodel)

	def fwdError(self,m,s_real,imodel=0):
		if(self.models[imodel].size() < 2): return 99999
		f = self.models[imodel]
		s_pred = f.forward_prediction(m)		
		err_fwd = np.linalg.norm(s_real-s_pred)**2
		return err_fwd

	def motorBabblingUpdate(self,m,s_real,imodel=0):
		err_fwd= self.fwdError(m,s_real,imodel=imodel)
		if(err_fwd < 99999): self.errLog(err_fwd,imodel)
		if(err_fwd > self.min_err):
			self.models[imodel].update(m,s_real)
			printProgress(self.models[imodel].size(),err_fwd, "%s-%d"%(self.strid,imodel))
			return True
		return False


	def goalBabblingUpdate(self,m,s_real,s_goal,imodel=0):
		err_inv = np.linalg.norm(s_real-s_goal)**2
		# much bigger errors because out of reach
		self.errLog(err_inv,imodel)
		self.im_goals[imodel].append(s_goal)
		self.im_models[imodel].update(np.hstack((m, s_goal)), np.hstack((m, s_real)))
		if(err_inv > self.min_err):
			self.models[imodel].update(m,s_real)
			printProgress(self.models[imodel].size(),err_inv, "%s-%d"%(self.strid,imodel))
			return True
		return False








			#inlimits = True
			# tin = min_max(tin, self.in_mins[imodel], self.in_maxs[imodel])
			# tout = min_max(tout, self.out_mins[imodel], self.out_maxs[imodel])
			# for i,v in enumerate(tin): inlimits = v >= self.in_mins[imodel][i] and v <= self.in_maxs[imodel][i] and inlimits
			# for i,v in enumerate(tout): inlimits = v >= self.out_mins[imodel][i] and v <= self.out_maxs[imodel][i] and inlimits 
			# if(not inlimits): 
			# 	print self.strid,"model",imodel,"err",err_fwd,err_inv, "NOT in LIMITS", 
			# 	for i,v in enumerate(tin):
			# 		print round(v,2),"(",self.in_mins[imodel][i],"..",self.in_maxs[imodel][i],")",
			# 	print ""

