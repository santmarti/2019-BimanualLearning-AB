import numpy as np
from numpy import linalg as npla
import random
import sys

import Box2DWorld
from Box2DWorld import TIME_STEP, vel_iters, pos_iters
import VectorFigUtils
import SensoriMotor
from SensoriMotor import RobotArmEnv
import ExplautoUtils

from explauto import Environment
from explauto import SensorimotorModel
from explauto import InterestModel
from explauto.interest_model.random import RandomInterest

import pickle
import os.path


def min_max(v,mins,maxs):
    v = np.minimum(v, maxs)
    v= np.maximum(v, mins)
    return v


class LearningModel():
    """ alfa_lim depends on the revoluteJoint limits that have now been lateralized
        default: -0.7 * 3.1415 ... 0.7 * 3.1415
        left arm: -0.85 * 3.1415 ... 0
        right arm: 0 ... 0.85 * 3.1415          """ 

    def iniModels(self):
        self.env = []
        self.in_mins, self.in_maxs, self.out_mins, self.out_maxs = [],[],[],[]
        self.matslist = []
        self.im_models = []
        self.models = []
        self.armModels(self.nparts, self.alfa_lim, self.bHand)
        self.armsDeltaRobotModel()
        self.armsNoDeltaRobotModel()
        self.twoArmsObjSensoryModel()

    def __init__(self, exp, nparts = 2, alfa_lim = (-2.7,2.7), bHand = False):
        print "Created Learning Model", "bDebug", Box2DWorld.bDebug
        self.exp = exp
        self.nao = exp.nao
        self.obj = exp.obj
        self.bHand = bHand
        self.nparts = nparts
        self.alfa_lim = alfa_lim
        self.folder = self.exp.name
        self.neval = 5
        self.iniModels()

        self.iniGoals()
        self.SM = SensoriMotor.SensoriMotorMem(exp)

        self.dm_lim,self.v_lim = exp.dm_lim, exp.v_lim
        self.strid = "learnModel"
        self.errorManager = ExplautoUtils.ErrorManager(self,folder='human')

        if(Box2DWorld.bDebug):
            self.SM.printModels("\nLearningModel",self.models,self.in_mins, self.in_maxs, self.out_mins, self.out_maxs)

    def getEnvModel(self, imodel=0):
        return self.env[imodel], self.models[imodel], self.im_models[imodel], self.matslist[imodel]


    def addModel(self,i): 
        self.env.append( RobotArmEnv(self.in_mins[i], self.in_maxs[i], self.out_mins[i], self.out_maxs[i]) )
        self.im_models.append( InterestModel.from_configuration(self.env[i].conf, self.env[i].conf.s_dims, 'discretized_progress') )
        self.models.append( SensorimotorModel.from_configuration(self.env[i].conf, 'nearest_neighbor') )
        self.matslist.append( [SensoriMotor.Matrix() for i in range(3)] )

    def addModelContext(self,i,context_mode,dummy=False): 
        nao = self.nao
        in_mins, in_maxs, out_mins, out_maxs = self.in_mins[i], self.in_maxs[i], self.out_mins[i], self.out_maxs[i]

        env_conf = dict(m_mins=in_mins,m_maxs=in_maxs,s_mins=out_mins,s_maxs=out_maxs)
        env_cls = SensoriMotor.RobotNaoEnv

        env = SensoriMotor.RobotArmEnvContext(env_cls, env_conf, context_mode, dummy=dummy)
        
        im_model =  RandomInterest(env.conf, env.conf.s_dims)
        model = SensorimotorModel.from_configuration(env.conf, 'nearest_neighbor')

        self.in_mins[i] += nao.dm_mins()
        self.in_maxs[i] += nao.dm_maxs()

        self.out_mins[i] += nao.ds_mins() # be careful with two end-points
        self.out_maxs[i] += nao.ds_maxs() 

        self.env.append(env)
        self.im_models.append(im_model); 
        self.models.append(model)
        self.matslist.append( [SensoriMotor.Matrix() for i in range(3)] )


    def armModels(self, nparts, alfa_lim, bHand = False):
        # f(m)=p
        x_lim, y_lim = VectorFigUtils.x_lim, VectorFigUtils.y_lim
        nRealParts = nparts
        if(bHand): nRealParts = nparts + 1 
        self.nRealParts = nRealParts
        for i in range(len(self.nao.arms)):  
            joint_lim = self.nao.getJointLimits(i)
            in_mins = [j[0] for j in joint_lim]
            in_maxs = [j[1] for j in joint_lim]
            self.in_mins.append( in_mins ) # nparts joints plus the hand
            self.in_maxs.append( in_maxs )
            self.out_mins.append( [x_lim[0],  y_lim[0]] )
            self.out_maxs.append( [x_lim[1],  y_lim[1]] )
            if(bHand):
                self.out_mins[i].append(-np.pi/2.0)
                self.out_maxs[i].append(np.pi/2.0)

            self.addModel(i)


    def armsDeltaRobotModel(self):
        # f(m,dm) = <s,s>
        x_lim, y_lim = VectorFigUtils.x_lim, VectorFigUtils.y_lim
        nao = self.nao
        dm_lim = self.exp.dm_lim
        if(len(nao.arms) < 2): return

        in_mins = []
        in_maxs = []
        for iarm in range(len(self.nao.arms)):
            joint_lim = nao.getJointLimits(iarm=iarm)
            in_mins += [j[0] for j in joint_lim]
            in_maxs += [j[1] for j in joint_lim]   # dm automatically added by the context

        context_mode = dict(choose_m=False,
                            rest_position=nao.rest_position(),
                            dm_bounds=[nao.dm_mins(),nao.dm_maxs()],
                            ds_bounds=[nao.ds_mins(),nao.ds_maxs()])

        out_mins = []
        out_maxs = [] 
        #for isal in range(len(nao.salient)):
        for isal in range(2):
            out_mins += [x_lim[0],  y_lim[0]] 
            out_maxs += [x_lim[1],  y_lim[1]] 

        self.in_mins.append( in_mins ) # nparts joints plus the hand
        self.in_maxs.append( in_maxs )
        self.out_mins.append( out_mins )
        self.out_maxs.append( out_maxs )

        self.addModelContext( len(self.models), context_mode )


    def armsNoDeltaRobotModel(self,iarms=[0,1]):
        # f(m) = <s,s>
        x_lim, y_lim = VectorFigUtils.x_lim, VectorFigUtils.y_lim
        nao = self.nao
        in_mins,in_maxs = [],[]
        if(len(nao.arms) < 2): return

        for iarm in iarms:
            joint_lim = nao.getJointLimits(iarm=iarm)
            in_mins += [j[0] for j in joint_lim]
            in_maxs += [j[1] for j in joint_lim]

        out_mins = [x_lim[0],  y_lim[0], x_lim[0],  y_lim[0]]
        out_maxs = [x_lim[1],  y_lim[1], x_lim[1],  y_lim[1]]
        
        self.in_mins.append( in_mins ) # nparts joints plus the hand
        self.in_maxs.append( in_maxs )
        self.out_mins.append( out_mins )
        self.out_maxs.append( out_maxs )

        self.addModel( len(self.models) )



    def twoArmsObjSensoryModel(self,iarms=[0,1]):  # f(pi,pj,vi,vj) = pobj,vobj
        x_lim, y_lim = VectorFigUtils.x_lim, VectorFigUtils.y_lim
        nao = self.nao
        dm_lim = self.exp.dm_lim

        in_mins = [x_lim[0],  y_lim[0], x_lim[0],  y_lim[0]]
        in_maxs = [x_lim[1],  y_lim[1], x_lim[1],  y_lim[1]]

        ds_mins,ds_maxs = [-1,-1], [1,1]

        context_mode = dict(choose_m=False,
                            rest_position=nao.rest_position(),
                            dm_bounds=[ds_mins*2,ds_maxs*2],
                            ds_bounds=[ds_mins,ds_maxs])

        out_mins = [x_lim[0],  y_lim[0]] 
        out_maxs = [x_lim[1],  y_lim[1]] 

        self.in_mins.append( in_mins ) # nparts joints plus the hand
        self.in_maxs.append( in_maxs )
        self.out_mins.append( out_mins )
        self.out_maxs.append( out_maxs )

        self.addModelContext( len(self.models), context_mode, dummy = True )



    def iniGoals(self):
        self.goals = [[]]*len(self.models)
        self.motor_pred = [[]]*len(self.models)

    def gotoDeltaS(self, s, ds, iarm = 0, online = True):
        spos = np.array(s.getxy()) + np.array(ds)
        self.goals[iarm] = spos
        m_pred = self.models[iarm].inverse_prediction(spos) 
        m_pred = min_max(m_pred, self.in_mins[iarm], self.in_maxs[iarm])
        self.motor_pred[iarm] = m_pred        
        if(online): self.nao.setTargetJoints(self.motor_pred[iarm], iarm=iarm)
        else: self.nao.gotoTargetJoints(self.motor_pred[iarm], iarm=iarm)

    def gotoS(self, iarm, s, online = True):
        self.goals[iarm] = s 
        m_pred = self.models[iarm].inverse_prediction(s) 
        m_pred = min_max(m_pred, self.in_mins[iarm], self.in_maxs[iarm])
        self.motor_pred[iarm] = m_pred        
        if(online): self.nao.setTargetJoints(self.motor_pred[iarm], iarm=iarm)
        else: self.nao.gotoTargetJoints(self.motor_pred[iarm], iarm=iarm)
        if(online == False): return self.nao.getFinalPos(iarm=iarm)

    def setObjTargetPos(self, d = 0.4, drand=[], bTarget=True):
        central = np.array([0,1.2])
        if(not drand): drand = 2*d*np.random.rand(2)-d
        if(bTarget): self.exp.setTargetObj(central + drand)
        self.exp.setObjPos(central - 2*drand)
        self.SM.reset()
        self.myWorldLoop(1)

    def setGrabTowardsTarget(self):
        sl = self.SM.getSalientObj(dir=-1)
        sr = self.SM.getSalientObj(dir=1)
        vl = self.SM.getTargetDir(dir=-1)
        vr = self.SM.getTargetDir(dir=1)
        self.gotoDeltaS(sl,vl,iarm=0)
        self.gotoDeltaS(sr,vr,iarm=1)

    def setGrabObjGoal(self,ini_far=0.32):
        pobj = self.exp.getObjPos() 
        for iarm in [0,1]:
            sign = 2*iarm - 1
            parm = self.nao.getFinalPos(iarm=iarm)
            h = self.exp.getFinalHaptic(iarm)
            do = ini_far
            pgrab = [pobj[0]+sign*do, pobj[1]] 
            d = VectorFigUtils.dist(parm,pgrab)
            d = min(d,2.0)/1.9
        
            if(h > 0.9): do *= 2.5
            if(d<0.1 and h < 0.05): do -= 2*d

            pgrab = [pobj[0]+sign*(do+d),pobj[1]] 

            self.goals[iarm] = pgrab
            m_pred = self.models[iarm].inverse_prediction(self.goals[iarm]) 
            m_pred = min_max(m_pred, self.in_mins[iarm], self.in_maxs[iarm])
            self.motor_pred[iarm] = m_pred        
            self.nao.setTargetJoints(self.motor_pred[iarm], iarm=iarm)

    def setSalientGoal(self, iarm = 0, isal = 0):
        snao = len(self.nao.salient)
        if(isal < snao): print "setSalientGoal", isal, "and nao has", snao    
        if(isal >= len(self.exp.salient)): print "isal",isal,"out of bounds"

        self.goals[iarm] = self.exp.salient[isal] 
        m_pred = self.models[iarm].inverse_prediction(self.goals[iarm]) 
        m_pred = min_max(m_pred, self.in_mins[iarm], self.in_maxs[iarm])
        self.motor_pred[iarm] = m_pred        

        if(iarm <= 1):
            self.nao.setTargetJoints(self.motor_pred[iarm], iarm=iarm)

    def getSalient(self,i):
        return self.SM.getSalient(i)

    def getM(self, imodel = 0, t=[]):
        if(t == []):
            iarm = imodel
            if(imodel > 1): iarm = imodel-2 
            print self.nao.getJointAngles(arm=iarm)
            t = self.nao.getJointAngles(arm=iarm)
            t += [0,0,0]

        pobj,pangle = self.exp.getObjPos(bAngle=True)
        parm = np.array(Box2DWorld.arm.getFinalPos())
        t[-3] = VectorFigUtils.vangle([1,0], np.array(pobj) - parm)
        t[-2] = pangle
        t[-1] = VectorFigUtils.dist(pobj,parm)
        return t

    def getRandomM(self, imodel = 0):
        t,nin = [], len(self.in_mins[imodel])
        for i in range(nin):
            r = self.in_maxs[imodel][i] - self.in_mins[imodel][i] 
            t += [r*np.random.rand()+self.in_mins[imodel][i]]
        return t

    def setSensoryGoals(self, imodel = 0, online = True):
        self.goals[imodel] = self.im_models[imodel].sample() 
        if(self.models[imodel].t > 5):  # first samples must be random
            t = self.models[imodel].inverse_prediction(self.goals[imodel]) 
            t = min_max(t, self.in_mins[imodel], self.in_maxs[imodel])
        else:
            t = self.getRandomM(imodel)

        if(imodel <= 1):   # usual arm model f(m)=s
            if(online): self.nao.setTargetJoints(t, arm=imodel)
            else:       self.nao.gotoTargetJoints(t, arm=imodel)
        elif(imodel <= 3): # combined arm/obj
            self.nao.gotoTargetJoints(t[:-3], arm=imodel-2)
            t=self.getM(imodel=imodel-2,t=t)

        t = [round(x,2) for x in t]
        self.motor_pred[imodel] = t
        return t


    def getSalientVel(self):
        fewSamples = SensoriMotor.Salient.vel_samples > self.SM.salientMap[0].newSamples
        if(fewSamples): return []
        else:
            return [s.getv() for i,s in self.SM.salientMap.iteritems()]


    def forward_error(self,m,s,iarm=0):
        fms = self.models[iarm]
        fms.mode = "exploit"
        s_pred = fms.forward_prediction(m)
        return VectorFigUtils.dist(s,s_pred)
    
    def forward_errors(self,iarm=0, probes = 10):
        errors = []
        env = self.env[iarm]
        arm = self.nao.arms[iarm]
        for m in env.random_motors(n=probes):
            s = np.array(arm.gotoTargetJoints(m))
            err = self.forward_error(m,s,iarm)
            errors.append(err)
        self.models[iarm].mode = "explore"
        return errors

    def learn(self):
        self.exp.setObjPos([-10,-10]) # out of business
        errors,mean_error,mean_var = [],[],[]
        for i in range(2):
            self.models[i].mode = 'explore'
            # gotoTargetJoints is a blocking all : we can only use it in training not in real time
            self.nao.restPosition(online=False)
            for j in range(3000):
                Box2DWorld.printProgress(j,self.models[i].size())
                t = self.env[i].getRandomInput()
                s = np.array(self.nao.gotoTargetJoints(t,iarm = i))
                m = self.nao.getJointAngles(iarm = i)
                err = 1
                if(self.models[i].size() > 1): err = self.forward_error(m,s,iarm=i)
                if(err > 0.03): self.models[i].update(m, s)
                self.exp.update()
                self.SM.update(self.exp)
                errors.append(err)
                if(len(errors) > 20): errors.pop(0)                
                #error = self.forward_errors()
                #VectorFigUtils.error_monitor(error,mean_error,mean_var)   
            print "\nLearn f(m,s) arms", i, "tuples:", self.models[i].size(), "error", np.mean(errors)
            self.save(i)
      

    def myWorldLoop(self, niters=100):
        nao = self.nao
        exp = self.exp
        SM = self.SM
        niter = 0
        err = exp.update()
        SM.update(exp)
        while(err > 0.05 and niter < niters):
            err = exp.update()
            SM.update(exp)
            Box2DWorld.world.Step(TIME_STEP, vel_iters, pos_iters)
            Box2DWorld.world.ClearForces()
            niter += 1

    def sampleSensoryMat(self,i):
        x_lim, y_lim = VectorFigUtils.x_lim, VectorFigUtils.y_lim
        x,y = self.SM.getSalient(i).sampleMat()
        x0,x1 = x_lim[0] - 1, x_lim[1] + 1
        y0,y1 = y_lim[0] - 0.5, y_lim[1] + 2
        s = (x0+x*(x1-x0),y0+y*(y1 - y0))
        return s
# ************************************
# Two Arms with Delta movements f(m,dm)=s,ds
# ***********************************

    def targets_sds(self, imodel=0, mode='mat', doit=False, online=True):
        env,model,im_model,mats = self.getEnvModel(imodel)
        nao = self.nao
        lm = len(env.random_motors(n=1)[0])/2

        tout = im_model.sample()

        if(imodel == 2):
            sds = tout
            ls = len(sds)/2
            s,ds = sds[:ls], sds[ls:]
        elif(imodel == 3):
            s = tout            

        if(mode=='mat'): s_left,s_right =  self.sampleSensoryMat(0), self.sampleSensoryMat(3)
        elif(mode=='interest'): s_left,s_right = s[:2],s[2:]

        s = np.hstack((s_left,s_right))

        model.mode = 'exploit'
        if(imodel == 2):
            sds = np.hstack((s,ds))
            if(doit):
                self.goals = [s_left,s_right]   # to draw targets on the screen
                mdm = model.inverse_prediction(sds)
                m = mdm[:lm]
                if(online): nao.setTargetJoints(m)
                else: nao.gotoTargetJoints(m)
                model.mode = 'explore'
            return sds
        elif(imodel == 3):
            if(doit):
                self.goals = [s_left,s_right]   # to draw targets on the screen
                m = model.inverse_prediction(s)
                if(online): nao.setTargetJoints(m)
                else: nao.gotoTargetJoints(m)
                model.mode = 'explore'
            return s
        model.mode = 'explore'
            

    def compute_ds(self,s_ini,s_fin):
        ds = np.array(s_fin) - np.array(s_ini)
        vnorm = npla.norm(ds)
        if(vnorm<=1): vnorm = 1 
        #return [round(ds[i]/vnorm,2) for i in range(len(ds))]
        return [round(ds[i],2) for i in range(len(ds))]

    def evaluateFwd_mdm(self,list_mdm,imodel=0):
        errMan = self.errorManager
        nao = self.nao
        f = self.models[imodel]
        f.mode = "exploit"
        lm = len(self.in_mins[imodel])/2
        ls = len(self.out_mins[imodel])/2
        errors = []
        for mdm in list_mdm:
            m,dm = mdm[:lm], mdm[lm:]
            sds_pred = f.forward_prediction(mdm)
            s_pred,ds_pred = sds_pred[:ls], sds_pred[ls:]
            nao.restPosition(online=False)
            s_real = nao.gotoTargetJoints(m)  
            s_after = nao.deltaMotorUpdate(dm=dm)
            ds = self.compute_ds(s_pred,s_after)
            errors.append( npla.norm( sds_pred - np.hstack((s_real,ds)))**2 )

        return [np.mean(errors), np.std(errors), np.min(errors), np.max(errors)]


    def evaluateInv_sds(self,list_sds,imodel=0):
        errMan = self.errorManager
        nao = self.nao
        f = self.models[imodel]
        l = len(self.in_mins[imodel])/2
        errors = []
        for sds in list_sds:    
            mdm = f.inverse_prediction(sds)
            m,dm = mdm[:l], mdm[l:]
            nao.restPosition(online=False)
            s_real = nao.gotoTargetJoints(m)  
            s_after = nao.deltaMotorUpdate(dm=dm)
            ds_real = self.compute_ds(s_real,s_after)
            errors.append( npla.norm(sds - np.hstack((s_real,ds_real))) )
        
        return [np.mean(errors), np.std(errors), np.min(errors), np.max(errors)]

    def evaluate_mdmsds(self,imodel=2):
        errMan = self.errorManager
        itest = 5
        env,model,im_model,mats = self.getEnvModel(imodel)
        model.mode = "exploit"
        if(model.size() > 0 and model.size()%itest == 0):
            list_mdm = env.random_motors(n=self.neval)
            list_sds = [self.targets_sds(imodel=imodel,mode='mat',doit=False) for _ in range(self.neval)]            
            eInv_mean,eInv_var,eInv_min,eInv_max = self.evaluateInv_sds(list_sds,imodel)
            eFwd_mean,eFwd_var,eFwd_min,eFwd_max = self.evaluateFwd_mdm(list_mdm,imodel)
            errMan.evaluation_hist[imodel].append( np.array([eInv_mean,eInv_var,eInv_min,eInv_max,eFwd_mean,eFwd_var,eFwd_min,eFwd_max]))
        model.mode = "explore"

    def dmGoalBabbling(self,imodel,mtrials=100,dmtrials=10):
        errMan = self.errorManager
        env,model,im_model,mats = self.getEnvModel(imodel)
        l = len(env.random_motors(n=1)[0])/2
        nao = self.nao
        conf = env.conf
        print "Goal Babbling :",
        for mtrials in range(mtrials):    
            sds = im_model.sample()
            if(model.size() > 0): mdm_inv = model.inverse_prediction(sds)
            else: mdm_inv = env.random_motors(n=1)[0]
            m_inv = mdm_inv[:l] 
        
            nao.restPosition(online=False)
            s = nao.gotoTargetJoints(m_inv)  
            for _ in range(dmtrials):
                model.mode = 'explore'
                m,s = nao.getJointAngles(), nao.getFinalPos()  
                ds_g = im_model.sample_given_context(s, range(conf.s_ndims/2))

                index_mfin = conf.m_ndims/2
                index_mdmfin = conf.m_ndims
                index_sdsfin = conf.m_ndims + conf.s_ndims

                in_dims = range(index_mfin) + range(index_mdmfin,index_sdsfin)
                out_dims = range(index_mfin, index_mdmfin)

                if(model.size() > 0): dm_pred = model.infer(in_dims, out_dims, np.hstack((m,s,ds_g)))
                else: dm_pred = env.random_motors(n=1)[0][l:]

                s_after = nao.deltaMotorUpdate(dm=dm_pred)
                ds = self.compute_ds(s,s_after)

                mdm,sds,sds_goal = np.hstack((m,dm_pred)), np.hstack((s,ds)), np.hstack((s,ds_g))
                bUpdate = errMan.goalBabblingUpdate(mdm,sds,sds_goal,imodel)
                self.evaluate_mdmsds(imodel=imodel)

        print "---> Goal Babbling Ended.   Model Size",model.size()



    def dmMotorBabbling(self,imodel,mtrials=100,dmtrials=10):
        env,model,im_model,mats = self.getEnvModel(imodel)
        nao = self.nao
        l = len(env.random_motors(n=1)[0])/2
        print "Motor Babbling :",
        for i,mdm in enumerate(env.random_motors(n=mtrials)):
            m = mdm[:l] 
            nao.restPosition(online=False)
            s = nao.gotoTargetJoints(m)  
            mats[0].add(s[0],s[1])
            m_now = nao.getJointAngles()
            for dm in env.random_dm(n=dmtrials):
                model.mode = 'explore'
                m = nao.getJointAngles()
                s = nao.getFinalPos() 
                s_after = nao.deltaMotorUpdate(dm=dm)
                ds = self.compute_ds(s,s_after)
                mats[1].add(s_after[0],s_after[1])
                tin,tout = np.hstack((m,dm)), np.hstack((s,ds))
                self.errorManager.motorBabblingUpdate(tin,tout,imodel)
                self.evaluate_mdmsds(imodel=imodel)
        print "---> Motor Babbling Ended.   Model Size",model.size()

    def learnDeltaArms(self,strid,nmotor=1,ndmmotor=1,nbabbling=1,ndmbabbling=1):  # delta only of the arms, not with object
        self.strid = strid
        nao = self.nao
        exp = self.exp
        SM = self.SM
        imodel = 2 # learnDeltaTwoArms without Object :  f(m,dm) = <s,s>
        env,model,im_model,mats = self.getEnvModel(imodel)

        print "learnDeltaTwoArms imodel", imodel," :  ", "f(m,dm) = <s_left,s_right>  model size: ", model.size()
        nao.restPosition(online=False)

        self.errorManager.min_err = 0.05
        self.dmMotorBabbling(imodel,mtrials=nmotor,dmtrials=ndmmotor)
        self.dmGoalBabbling(imodel,mtrials=nbabbling,dmtrials=ndmbabbling)
        self.save(imodel=imodel)


# ************************************
# Two Arms No Delta f(m)=s
# ************************************

    def evaluateFwd_m(self,list_m,imodel=0):
        errMan = self.errorManager
        nao = self.nao
        f = self.models[imodel]
        errors = []
        for m in list_m:
            s_pred = f.forward_prediction(m)
            nao.restPosition(online=False)
            s_real = nao.gotoTargetJoints(m)  
            errors.append( npla.norm(s_pred - s_real)**2 )
        return [np.mean(errors), np.std(errors), np.min(errors), np.max(errors)]


    def evaluateInv_s(self,list_s,imodel=0):
        errMan = self.errorManager
        nao = self.nao
        f = self.models[imodel]
        errors = []
        for s in list_s:
            m = f.inverse_prediction(s)
            nao.restPosition(online=False)
            s_real = nao.gotoTargetJoints(m) 
            m_real =  nao.getJointAngles()
            errors.append( npla.norm( s-s_real )**2 )
        return [np.mean(errors), np.std(errors), np.min(errors), np.max(errors)]

    def evaluate_ms(self,imodel=3):
        errMan = self.errorManager
        itest = 5
        env,model,im_model,mats = self.getEnvModel(imodel)
        model.mode = "exploit"
        if(model.size() > 0 and model.size()%itest == 0):
            list_m = env.random_motors(n=self.neval)
            list_s = [self.targets_sds(imodel=3,mode='mat',doit=False) for _ in range(self.neval)]            
            eInv_mean,eInv_var,eInv_min,eInv_max = self.evaluateInv_s(list_s,imodel)
            eFwd_mean,eFwd_var,eFwd_min,eFwd_max = self.evaluateFwd_m(list_m,imodel)
            errMan.evaluation_hist[imodel].append( np.array([eInv_mean,eInv_var,eInv_min,eInv_max,eFwd_mean,eFwd_var,eFwd_min,eFwd_max]))
        model.mode = "explore"


    def goalBabbling(self,imodel,mtrials=100): # independant of the model
        errMan = self.errorManager
        env,model,im_model,mats = self.getEnvModel(imodel)
        nao = self.nao
        conf = env.conf
        print "Goal Babbling :",
        for mtrials in range(mtrials):    
            s_goal = im_model.sample()

            if(model.size() > 1): m_inv = model.inverse_prediction(s_goal)
            else: m_inv = env.random_motors(n=1)[0]
            nao.restPosition(online=False)

            s_real = nao.gotoTargetJoints(m_inv)  
            m_now = nao.getJointAngles()

            bUpdate = errMan.goalBabblingUpdate(m_now,s_real,s_goal,imodel)
            self.evaluate_ms(imodel=imodel)
        print "---> Goal Babbling Ended.   Model Size",model.size()


    def motorBabbling(self,imodel,mtrials=100):   # only motor independent of num arms
        env,model,im_model,mats = self.getEnvModel(imodel)
        nao = self.nao
        l = len(env.random_motors(n=1)[0])/2
        print "Motor Babbling :",
        for i,m in enumerate(env.random_motors(n=mtrials)):
            nao.restPosition(online=False)
            s_real = nao.gotoTargetJoints(m)  
            m_now = nao.getJointAngles()
            self.errorManager.motorBabblingUpdate(m,s_real,imodel)
            self.evaluate_ms(imodel=imodel)

        print "---> Motor Babbling Ended.   Model Size",model.size()

    def learnNoDeltaArms(self,strid, nmotor=1,nbabbling=1):  # delta only of the arms, not with object
        self.strid = strid
        nao = self.nao
        exp = self.exp
        SM = self.SM
        imodel = 3 # learnNoDeltaTwoArms :  f(m) = <s,s>
        env,model,im_model,mats = self.getEnvModel(imodel)

        print "learnNoDeltaTwoArms imodel", imodel," :  ", "f(m) = <s_left,s_right>  model size: ", model.size()
        nao.restPosition(online=False)
        self.errorManager.min_err = 0.05
        self.motorBabbling(imodel,mtrials=nmotor)
        self.goalBabbling(imodel,mtrials=nbabbling)
        self.save(imodel=imodel)

# ************************************
# END -------- Two Arms No Delta
# ************************************




# *********************************************
# Two Arms No Delta Obj f(pi,pj,vi,vj)=pobj,vj
# *********************************************


    def goalObjBabbling(self,imodel,mtrials=100): # independant of the model
        errMan = self.errorManager
        env,model,im_model,mats = self.getEnvModel(imodel)
        nao = self.nao
        conf = env.conf
        print "Goal Babbling :",
        for mtrials in range(mtrials):    
            s = im_model.sample()

            m_inv = model.inverse_prediction(s)
            m_inv = SensoriMotor.bounds_min_max(m_inv,self.in_mins[imodel],self.in_maxs[imodel]) 
            nao.restPosition(online=False)

            print "before",m_inv
            s_real = nao.gotoTargetJoints(m_inv)  
            m_now = nao.getJointAngles()

            bUpdate = errMan.errorUpdate(m_now,s_real,imodel)
            im_model.update(np.hstack((m_inv, s)), np.hstack((m_inv, s_real)))
            #self.evaluate_ms(imodel=imodel)
        print "---> Goal Babbling Ended.   Model Size",model.size()


    def motorObjBabbling(self,imodel,mtrials=100):   # only motor independent of num arms
        env,model,im_model,s = self.getEnvModel(imodel)
        nao = self.nao
        l = len(env.random_motors(n=1)[0])/2
        print "Motor Babbling :",
        for i,ssdsds in enumerate(env.random_motors(n=mtrials)):
            nao.restPosition(online=False)
            exp.setSensoryGoals()
            s_real = nao.gotoTargetJoints(m)  
            m_now = nao.getJointAngles()
            err_reach = sum(abs(np.array(m)-np.array(m_now))) # reaching error 
            self.errorManager.errorUpdate(m,s_real,imodel)
            #self.evaluate_ms(imodel=imodel)

        print "---> Motor Babbling Ended.   Model Size",model.size()

    def learnTwoArmsObj(self,strid, nmotor=1,nbabbling=1):  
        # f(pi,pj,vi,vj) = pobj,vobj
        self.strid = strid
        nao = self.nao
        exp = self.exp
        SM = self.SM
        imodel = 4 
        env,model,im_model = self.getEnvModel(imodel)

        print "twoArms imodel", imodel," :  ", "f(pi,pj,vi,vj) = pobj,vobj   model size: ", model.size()
        nao.restPosition(online=False)
        self.errorManager.min_err = 0.05
        self.motorObjBabbling(imodel,mtrials=nmotor)
        self.goalObjBabbling(imodel,mtrials=nbabbling)
        self.save()


# *********************************************
# END --------- Two Arms No Delta Obj f(pi,pj,vi,vj)=pobj,vj
# *********************************************


    def learnObjectPlay(self):
        import sys
        nao = self.nao
        exp = self.exp
        SM = self.SM
        ini_obj_pos = exp.ini_obj_pos

        print "learnObjectPlay -----------------------------------------"
        for iarm in range(2):
            print "IARM",iarm
            for itrial in range(1000):
                obj_pos = np.array(ini_obj_pos) + (0,1)
                exp.setObjPos(obj_pos, 0)

                print itrial,
                sys.stdout.flush()

                SM.reset()
                nao.setTargetJoints([2,0],arm=0)
                nao.setTargetJoints([-2,0],arm=1) 
                self.myWorldLoop()

                d = 1.2 # restart distance of ini_obj every 10 trials
                obj_pos = ini_obj_pos + d*np.random.rand(2) - d/2.0
                obj_angle = np.random.rand()*2*np.pi - np.pi
                exp.setObjPos(obj_pos, obj_angle)

                for j in range(10):
                    alfa = np.random.rand()*2*np.pi - np.pi 
                    v = np.array(VectorFigUtils.vrotate([1,0],alfa))
                    iSalObj = len(nao.getSalient())
                    target = np.array( exp.salient[iSalObj] ) + 0.3*v
                    self.gotoS(imodel=iarm, s=target, online=True)
                    self.myWorldLoop()
            print "\n"

        SM.save()



    def iniExp(self,folder,strid):
        if(folder == ""): folder = self.exp.name
        self.folder, self.strid = folder, strid
        self.iniModels()
        self.errorManager = ExplautoUtils.ErrorManager(self,folder=folder)
        self.errorManager.strid=strid
        self.SM.folder = folder
        self.SM.load(0)
        self.SM.load(3)

    def save_model(self,imodel):
        strid, folder = self.strid, self.folder
        i = imodel
        lms = self.matslist[i]
        for j,m in enumerate(lms):
            fname = "data/%s/%s-matlearn%d-%d.data"%(folder,strid,i,j)
            np.save(fname, m)   
        fname = "data/%s/%s-model%d.data"%(folder,strid,i)
        pickle.dump(self.models[i], open(fname, "wb"))
        fname = "data/%s/%s-im_model%d.data"%(folder,strid,i)
        pickle.dump(self.im_models[i], open(fname, "wb"))

    def save(self,imodel=-1):
        print "save() Robot Learning: ",imodel,self.strid
        if(imodel<0):
            for i in range(len(self.models)): 
                self.save_model(i)
        else: self.save_model(imodel)

    def load(self,folder="",strid=""):
        if(folder!=""): self.folder = folder 
        else: folder = self.folder
        if(strid == ""): strid = self.strid
        #bDebug = Box2DWorld.bDebug
        bDebug = True
        if(bDebug): print "Loading RobotLearning Models: ", len(self.models), "folder", folder, "strid", strid

        if(strid!=""):
            for i in range(len(self.models)):
                fname = "data/%s/%s-model%d.data"%(folder,strid,i)
                if(os.path.isfile(fname)):
                    if(bDebug): print fname, 
                    self.models[i] = pickle.load(open(fname, "rb"))
                    if(bDebug): print "with",self.models[i].size(),"tuples"

            for i in range(len(self.im_models)):
                fname = "data/%s/%s-im_model%d.data"%(folder,strid,i)
                if(os.path.isfile(fname)):
                    if(bDebug): print fname, 
                    self.im_models[i] = pickle.load(open(fname, "rb"))
                    if(bDebug): print "with..."  #,self.im_models[i].size(),"tuples"

        self.SM.load()
        self.SM.reset()



