import time
import numpy as np
from numpy import linalg as npla
from scipy.ndimage.filters import gaussian_filter
import math
import PyGameUtils
import VectorFigUtils
import Box2DWorld
import ExplautoUtils
from explauto import Environment
from explauto import SensorimotorModel
from explauto import InterestModel
import pickle
import os.path

def rand_bounds(bounds, n=1):
    widths = np.tile(bounds[1, :] - bounds[0, :], (n, 1))
    return widths * np.random.rand(n, bounds.shape[1]) + np.tile(bounds[0, :], (n, 1))

def bounds_min_max(v, mins, maxs):
    res = np.minimum(v, maxs)
    res = np.maximum(res, mins)
    return res


class RobotArmEnv(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

    def compute_motor_command(self, joint_pos_ag):
        return bounds_min_max(joint_pos_ag, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, joint_pos_env):
        return Box2DWorld.arm.gotoTargetJoints(joint_pos_env)
    
    def plot(self, ax, m):
        Box2DWorld.arm.gotoTargetJoints(m)
        Box2DWorld.plotWorld(ax)

    def getRandomInput(self):
        m_mins = self.conf.m_mins
        m_maxs = self.conf.m_maxs
        l = len(m_mins)
        m = [round(r*(m_maxs[i]-m_mins[i])+m_mins[i],2) for i,r in enumerate(np.random.rand(l))]
        return m
        
    def getRandomOutput(self):
        s_mins = self.conf.s_mins
        s_maxs = self.conf.s_mins
        l = len(s_mins)
        s = [r*(s_maxs[i]-s_mins[i])+s_mins[i] for i,r in enumerate(np.random.rand(l))]
        return s


class RobotNaoEnv(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

    def compute_motor_command(self, joint_pos_ag):
        return bounds_min_max(joint_pos_ag, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, joint_pos_env):
        return Box2DWorld.nao.gotoTargetJoints(joint_pos_env)
    
    def plot(self, ax, m):
        Box2DWorld.nao.gotoTargetJoints(m)
        Box2DWorld.plotWorld(ax)

    def getRandomInput(self):
        m_mins = self.conf.m_mins
        m_maxs = self.conf.m_maxs
        l = len(m_mins)
        m = [round(r*(m_maxs[i]-m_mins[i])+m_mins[i],2) for i,r in enumerate(np.random.rand(l))]
        return m
    def getRandomOutput(self):
        s_mins = self.conf.s_mins
        s_maxs = self.conf.s_mins
        l = len(s_mins)
        s = [r*(s_maxs[i]-s_mins[i])+s_mins[i] for i,r in enumerate(np.random.rand(l))]
        return s


class RobotArmEnvContext(Environment):
    def __init__(self, env_cls, env_conf, context_mode, dummy=False):
        self.rest_position = context_mode['rest_position']
        self.context_mode = context_mode        
        self.env = env_cls(**env_conf)          
        self.dummy = dummy
        
        Environment.__init__(self, 
                            np.hstack((self.env.conf.m_mins, self.context_mode['dm_bounds'][0])), 
                            np.hstack((self.env.conf.m_maxs, self.context_mode['dm_bounds'][1])),
                            np.hstack((self.env.conf.s_mins, self.context_mode['ds_bounds'][0])),
                            np.hstack((self.env.conf.s_maxs, self.context_mode['ds_bounds'][1])))                                            
        self.reset()

    def compute_motor_command(self, ag_state):
        return bounds_min_max(ag_state, self.conf.m_mins, self.conf.m_maxs)
    
    def compute_sensori_effect(self, mdm):
        l = len(mdm)/2
        m,dm = mdm[:l],mdm[l:]
        if self.context_mode['choose_m'] == True: 
            self.env.update(m, reset=False) 
            s = np.array(Box2DWorld.arm.gotoTargetJoints(m))
            self.current_motor_position = m
            self.current_sensori_position = s
        else:
            self.current_sensori_position = np.array(Box2DWorld.arm.getFinalPos())
            s = self.current_sensori_position

        snew = np.array(Box2DWorld.arm.deltaMotorUpdate(dm))
        self.current_motor_position = np.array(Box2DWorld.arm.getJointAngles())
        self.env.update(self.current_motor_position, reset=False)
        self.current_sensori_position = snew
        ds = snew - s
        return np.hstack((s,ds))

    def random_m(self, n=1):
        return rand_bounds(self.conf.bounds[:, len(self.conf.m_dims):len(self.conf.m_dims)/2], n)

    def random_dm(self, n=1):
        return rand_bounds(self.conf.bounds[:, len(self.conf.m_dims)/2:len(self.conf.m_dims)], n)

    def reset(self):
        if(not self.dummy):
            self.current_motor_position = np.array(self.rest_position)
            self.current_sensori_position = np.array(self.env.update(self.current_motor_position, reset=True))

    def plot(self, ax, **kwargs):
        alpha = 0.3
        if('alpha' in kwargs.keys()): alpha = kwargs['alpha']
        Box2DWorld.plotWorld(ax,alpha=alpha)





class Matrix():

    @staticmethod
    def makeGaussian(size, fwhm = 5, center=None):
        """ Make a square gaussian kernel: size is the length of a side of the square
        fwhm is full-width-half-maximum, which can be thought of as an effective radius."""
        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]
        if center is None: x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

    gaussian_kernel = makeGaussian.__func__(32)

    def __init__(self,xlim=[],ylim=[]):
        self.w,self.h = 640,480 # found in PyGameUtils but hardcoded to make them indep
        if(xlim is []):
            self.xlim, self.ylim = VectorFigUtils.x_lim, VectorFigUtils.y_lim
        else:
            self.xlim, self.ylim = xlim,ylim
        self.m = np.zeros((self.w,self.h))

    def add(self, x, y):
        xlim = self.xlim
        ylim = self.ylim
        i0 = (x - xlim[0]) / float(xlim[1]-xlim[0])
        j0 = (y - ylim[0]) / float(ylim[1]-ylim[0])

        n = len(Matrix.gaussian_kernel[0])
        i = i0*self.w - n/2
        j = self.h - j0*self.h - n/2

        for row1, row2 in zip(self.m[i:], Matrix.gaussian_kernel):
            row = row1[j:n + j]
            row1[j:n + j] = map(sum, zip(row2, row))
            
            max_val = 100
            indices = row1 > max_val
            row1[indices] = max_val

    def sampleMat(self):
        maux = np.copy(self.m)
        maux = maux / np.max(maux)
        w,h = maux.shape
        th = 0.01
        niter, bFound = 0, False
        while not bFound:
            i,j = np.random.randint(w), np.random.randint(h)
            bFound = (maux[i,j] > th)
            niter += 1
            if(niter%100==0): th -= 0.01
        return (i/float(w),1-j/float(h))

class Salient():

    vel_samples  = 8
    min_velocity = 0.2
    size_history  = 500

    def __init__(self,i,exp,mytype):
        self.id = i
        self.strid = "%s%d"%('s',i)

        self.history = []
        self.mat = Matrix()

        self.type = mytype
        self.vel = [0,0]
        self.bIncr = False
        self.bDecr = False
        self.newSamples = 0
        self.v_lim = exp.v_lim
        self.dm_lim = exp.dm_lim

        self.env = []
        self.im_models = []
        self.models = []
        self.in_mins, self.in_maxs, self.out_mins, self.out_maxs = [],[],[],[]

        numNaoSalient = len(exp.nao.salient)

        if(self.type == "left"): 
            #self.addArmModel(exp.nao,0)
            self.addDeltaModel(exp.nao,iarms=[0])
            self.addBlind(exp.nao)

        if(self.type == "right"): 
            #self.addArmModel(exp.nao,1)
            self.addDeltaModel(exp.nao,iarms=[1])
            self.addBlind(exp.nao)

        if(self.type == "obj"): 
            for i in range(numNaoSalient): 
                self.addObjModel()

            #if(self.id == numNaoSalient):
            #for i in range(numNaoSalient): 
            #    self.addObjTwoPoints()

            #self.addDeltaModel(exp.nao)
            self.addBimanualModel(exp.nao)
            #self.addBimanualModelNoAngle(exp.nao)

        self.goals={}
        for i in range(len(self.im_models)): self.goals[i]=[]
        self.errorManager = ExplautoUtils.ErrorManager(self,folder='human/salient')


    def addModel(self,i): 
        self.env.append( RobotArmEnv(self.in_mins[i], self.in_maxs[i], self.out_mins[i], self.out_maxs[i]) )
        self.im_models.append( InterestModel.from_configuration(self.env[i].conf, self.env[i].conf.s_dims, 'discretized_progress') )
        self.models.append( SensorimotorModel.from_configuration(self.env[i].conf, 'nearest_neighbor') )

    def addArmModel(self,nao,iarm): 
        #f(m)=pi   part of m is redundant depending on s
        #          better to learn al si at once
        x_lim, y_lim = VectorFigUtils.x_lim, VectorFigUtils.y_lim
        i = len(self.in_mins)
        joint_lim = nao.getJointLimits(iarm=iarm)
        in_mins = [j[0] for j in joint_lim]
        in_maxs = [j[1] for j in joint_lim]
        self.in_mins.append( in_mins ) 
        self.in_maxs.append( in_maxs )
        self.out_mins.append( [x_lim[0],  y_lim[0]] )
        self.out_maxs.append( [x_lim[1],  y_lim[1]] )
        self.addModel(i)

    def addDeltaModel(self,nao,iarms=[0,1]): 
        #f(m,dm)=vi,pi    -> m of one arm iarms=[0]
        #                 -> m of two arms iarms=[0,1]
        x_lim, y_lim = VectorFigUtils.x_lim, VectorFigUtils.y_lim
        dm_lim = self.dm_lim
        v_lim = self.v_lim
               
        for iarm in iarms:
            joint_lim = nao.getJointLimits(iarm=iarm)
            in_mins = [j[0] for j in joint_lim]
            in_maxs = [j[1] for j in joint_lim]
        for iarm in iarms:
            joint_lim = nao.getJointLimits(iarm=iarm)
            in_mins += [-dm_lim for j in joint_lim]
            in_maxs += [dm_lim for j in joint_lim]

        i = len(self.in_mins)
        self.in_mins.append(in_mins)
        self.in_maxs.append(in_maxs)
        self.out_mins.append( [-v_lim, -v_lim, x_lim[0],  y_lim[0]] )
        self.out_maxs.append( [v_lim, v_lim, x_lim[1],  y_lim[1]] )
        self.addModel(i)

    def addObjModel(self): #f(pi,vi,pobj)=vobj,pobj   ---  tin : pj,vj,pi  tout = vi,pi
        x_lim, y_lim = VectorFigUtils.x_lim, VectorFigUtils.y_lim
        v_lim = self.v_lim
        i = len(self.in_mins)

        self.in_mins.append( [x_lim[0],  y_lim[0], -v_lim, -v_lim, x_lim[0],  y_lim[0]] )
        self.in_maxs.append( [x_lim[1],  y_lim[1], v_lim, v_lim, x_lim[1],  y_lim[1]] )
        self.out_mins.append( [-v_lim, -v_lim, x_lim[0],  y_lim[0]] )
        self.out_maxs.append( [v_lim, v_lim, x_lim[1],  y_lim[1]] )
        self.addModel(i)

    def addObjTwoPoints(self):
        #f(pi,vi,pobj1,pobj2)=vobj1,vobj2,pobj1,pobj2
        x_lim, y_lim = VectorFigUtils.x_lim, VectorFigUtils.y_lim
        v_lim = self.v_lim
        in_mins = [x_lim[0],  y_lim[0]]
        in_maxs = [x_lim[1],  y_lim[1]]
        in_mins += [-v_lim for j in range(2)]
        in_maxs += [v_lim for j in range(2)]
        in_mins += [x_lim[0],  y_lim[0],x_lim[0],  y_lim[0]]
        in_maxs += [x_lim[1],  y_lim[1],x_lim[1],  y_lim[1]]

        i = len(self.in_mins)
        self.in_mins.append(in_mins)
        self.in_maxs.append(in_maxs)
        self.out_mins.append( [-v_lim, -v_lim, -v_lim, -v_lim, x_lim[0],  y_lim[0],  x_lim[0],  y_lim[0]] )
        self.out_maxs.append( [v_lim, v_lim, v_lim, v_lim, x_lim[1],  y_lim[1], x_lim[1],  y_lim[1]] )
        self.addModel(i)
       

    def addBimanualModel(self,nao): #f(vi,vj)=vobj,pobj   
        x_lim, y_lim = VectorFigUtils.x_lim, VectorFigUtils.y_lim
        v_lim = self.v_lim
        in_mins = [-v_lim, -v_lim, -v_lim, -v_lim] # vi,vj,alfa_rot
        in_maxs = [v_lim, v_lim, v_lim, v_lim] # vi,vj,alfa_rot

        i = len(self.in_mins)
        self.in_mins.append(in_mins)
        self.in_maxs.append(in_maxs)
        self.out_mins.append( [-v_lim, -v_lim, -np.pi/4.0, x_lim[0],  y_lim[0]] )
        self.out_maxs.append( [v_lim, v_lim, np.pi/4.0, x_lim[1],  y_lim[1]] )
        self.addModel(i)
        self.index_bimanual = i 

    def addBimanualModelNoAngle(self,nao): #f(vi,vj)=vobj,pobj   
        x_lim, y_lim = VectorFigUtils.x_lim, VectorFigUtils.y_lim
        v_lim = self.v_lim
        in_mins = [-v_lim, -v_lim, -v_lim, -v_lim] # vi,vj
        in_maxs = [v_lim, v_lim, v_lim, v_lim] # vi,vj

        i = len(self.in_mins)
        self.in_mins.append(in_mins)
        self.in_maxs.append(in_maxs)
        self.out_mins.append( [-v_lim, -v_lim, x_lim[0],  y_lim[0]] )
        self.out_maxs.append( [v_lim, v_lim, x_lim[1],  y_lim[1]] )
        self.addModel(i)
        self.index_bimanual = i 


    def addBlind(self,nao): #f(pi,vi)=pobj1,pobj2   when hi>0
        x_lim, y_lim = VectorFigUtils.x_lim, VectorFigUtils.y_lim
        v_lim = self.v_lim
        in_mins = [x_lim[0],  y_lim[0], -v_lim, -v_lim]
        in_maxs = [x_lim[1],  y_lim[1], v_lim, v_lim]

        i = len(self.in_mins)
        self.in_mins.append(in_mins)
        self.in_maxs.append(in_maxs)
        self.out_mins.append( [x_lim[0],  y_lim[0], x_lim[0],  y_lim[0]] )
        self.out_maxs.append( [x_lim[1],  y_lim[1], x_lim[1],  y_lim[1]] )
        self.addModel(i)
        #add one tuple so that we can bootstrap
        self.models[i].update([0.]*len(in_mins), np.hstack(([0., 0.],[0.,2])))


    def reset(self):
        self.bIncr = False
        self.bDecr = False
        self.vel = [0,0]
        self.newSamples = 0

    def getxy(self): # history tuple [x,y,vx,vy,h] 
        return self.history[-1][0:2]

    def geth(self):  # history tuple [x,y,vx,vy,h]
        #print self.id, self.type, self.history[-1]
        return self.history[-1][-1]

    def getvnorm(self):  
        return npla.norm(self.vel)

    def getvnormalized(self):
        n = npla.norm(self.vel)  
        if(n == 0): return [0,0]
        return [self.vel[0]/n, self.vel[1]/n]


    def getv(self):  
        return self.vel

    def computeVelFromHist(self):
        hsubs = np.array(self.history[-1])-np.array(self.history[-Salient.vel_samples])
        gain = 5
        vel = (round(gain*hsubs[0],2), round(gain*hsubs[1],2))
        vnorm = npla.norm(vel)
        if(vnorm>1): vel = (round(vel[0]/vnorm,2),round(vel[1]/vnorm,2))
        return vel

    def inverse(self,i,tin,tout):
        #print "Prediction model tuples ",self.models[i].t
        #print "tin:",tin
        #print "tout:",tout
        t = np.concatenate( (tin[0:2],(2,0),tout[2:4]), axis=0 )
        #print "t:",t
        #print self.models[i].forward_prediction(tin)
        pred = self.models[i].infer([4,5,6,7,8,9],[0,1,2,3],t)
        #pred = self.models[i].infer([0,1,2,3,4,5,8,9],[6,7],t)
        return pred 

    def isCausalContext(self,s):
        isdiff = s.id != self.id 
        isarm = s.type in ["left","right"]
        istouching = s.geth() > 0.3
        return isdiff and isarm and istouching

    def getInOut(self,exp,s):  #tin = pj,vj,pi  tout = vi,pi   
        p = s.getxy()
        v = s.getv()
        sp = self.getxy()                
        sv = self.getv()
        tin = [p[0],p[1],v[0],v[1],sp[0],sp[1]]
        tout = [sv[0],sv[1],sp[0],sp[1]]
        tin = [round(e,2) for e in tin]
        tout = [round(e,2) for e in tout]
        return tin,tout
 


    def isArm(self):
        if(self.type == "left"): return 0
        elif(self.type == "right"): return 1
        else: return -1 
    

    def updateModels(self,SM,exp):
        nao =  exp.nao
        p = self.getxy()
        v = self.getv()

        sobj, sobj1 = SM.getSalientObj(), SM.getSalientObj(next=1)
        pobj,vobj = sobj.getxy(), sobj.getv()
        pobj1,vobj1 = sobj1.getxy(), sobj1.getv()

        iarm = self.isArm()
        bUpdate = False

        if(iarm >= 0):
            m = nao.getJointAngles(iarm = iarm)
            dm = exp.getMotorHistory(iarm = iarm,t=-2)

            tin, tout = np.hstack((m,dm)), np.hstack((v,p))
            tin= bounds_min_max(tin,self.in_mins[0],self.in_maxs[0])
            bUpdate = bUpdate or self.errorManager.errorUpdate(tin,tout,0)

            tin, tout = np.hstack((p,v)), np.hstack((pobj,pobj1))
            bUpdate = bUpdate or self.errorManager.errorUpdate(tin,tout,1)

        elif(self.type == "obj"):
            for i,s in SM.salientMap.iteritems():
                if self.isCausalContext(s):
                    tin,tout = self.getInOut(exp,s)
                    bUpdate = bUpdate or self.errorManager.errorUpdate(tin,tout,s.id)

                    tin = np.hstack((p,v,pobj,pobj1))
                    tout = np.hstack((vobj,vobj1,pobj,pobj1))
                    bUpdate = bUpdate or self.errorManager.errorUpdate(tin,tout,s.id+sobj.id)

        if(bUpdate): self.mat.add(p[0],p[1])


    def isEvent(self):
        vnorm = self.getvnorm()
        if(not self.bIncr and vnorm >= Salient.min_velocity): 
            self.bIncr, self.bDecr = True,False    
            #if(self.id==4): print ">salient",self.id,"moving", vel
            return True
        elif(not self.bDecr and vnorm <= Salient.min_velocity):
            self.bIncr, self.bDecr = False,True            
            #if(self.id==4): print ">salient",self.id,"stopping",vel
            return False
        return False

    def addHistory(self,exp):
        x,y = exp.salient[self.id]
        h = exp.haptic[self.id]        
        hist = np.array([x,y,self.vel[0],self.vel[1],self.getvnorm(),h])
        hist = [round(e,2) for e in hist]
        self.history.append(hist)
        if(len(self.history) > Salient.size_history): 
            self.history.pop(0)     
        self.newSamples += 1

    def update(self,SM,exp):
        isEvent = False

        if(Salient.vel_samples <= self.newSamples): self.vel = self.computeVelFromHist()
        else: self.vel = [0,0] 

        self.addHistory(exp)
        if(Salient.vel_samples <= self.newSamples):
            if(self.isEvent()): 
                isEvent = True
                #self.updateModels(SM,exp)        

        return isEvent

    def getMat(self):  return self.mat.m

    def sampleMat(self):
        mat = self.mat
        x,y = mat.sampleMat()
        return (x,y)


class SensoriMotorMem():

    def __init__(self, exp):
        bDebug = Box2DWorld.bDebug
        self.exp, self.nao, self.obj = exp, exp.nao, exp.obj        
        narms = len(self.nao.arms)
        self.v_lim = exp.v_lim
        self.dm_lim = exp.dm_lim
        self.folder = self.exp.name

        if(bDebug): print "\nSalient points models:\n"
        self.salientMap = {}
        for i in range( len(exp.salient) ): 
            S = Salient(i,self.exp,exp.getSalientType(i))
            self.salientMap[i] = S
            if(bDebug): self.printModels("%s s%d"%(S.type,i), S.models, S.in_mins, S.in_maxs, S.out_mins, S.out_maxs)
        if(bDebug): print "Total:",len(self.salientMap),"points"


    def existsMap(self,id):
        if(id in self.salientMap): return True
        else: return False


    def sampleMat(self,id):
        if(id not in self.salientMap): return []
        s = self.salientMap[id]
        x,y = s.sampleMat()
        return (x,y)

    def setObjPos(self,p=[],angle = 0):
        self.exp.setObjPos(p,angle)
        self.reset()

    def getTargetDir(self,dir=0):
        exp = self.exp
        pt = np.array(exp.target_obj.position)
        dt = np.array(pt)-np.array(self.obj.position)
        nt = VectorFigUtils.vnorm(dt)
        if(dir == 0): return dt
        else:
            spos = self.getSalientObj(dir=dir).getxy()
            if(nt < 0.2): spos = np.array(self.obj.position) 
            d = np.array(pt)-np.array(spos)
            #if(VectorFigUtils.vnorm(d)>1): d /= 1.8             
        return d

    def getSalientObj(self, dir=0, next=0):
        sobj = self.getSalient(0)
        for i,s in self.salientMap.iteritems(): 
            if(s.type == "obj"): 
                sobj = s
                if(next==0): break
                else: next -= 1

        if(dir==0): return sobj  

        obj_pos = np.array(sobj.getxy()) + np.array([0.5*dir,0])        
        dmin,smin = 9999, sobj
        for i,s in self.salientMap.iteritems(): 
            if(s.type == "obj"): 
                d = VectorFigUtils.dist(obj_pos,s.getxy())
                if(d < dmin): dmin,smin = d,s
        return smin

    def getSalientRight(self): 
        nleft = len(self.nao.arms[0].salient)
        return self.salientMap[nleft]

    def getSalientLeft(self): return self.salientMap[0]


    def getSalient(self,id):
        if(id in self.salientMap): return self.salientMap[id]
        else: print "No salientMap with id", id 

    def getHistory(self,id):
        if(id in self.salientMap): return np.array(self.salientMap[id].history)
        else: print "No salientMap with id", id 

    def getStateMat(self,id):
        if(id in self.salientMap): return self.salientMap[id].getMat()
        else: print "No salientMap with id", id 


    def update(self,exp):
        isEvent = False
        positions = exp.getSalient()
        for i,p in enumerate(positions): 
            isEvent = isEvent or self.salientMap[i].update(self,exp)

        sec = int(time.time()- Box2DWorld.start_time)

        if(sec > 0 and sec % 1000 == 0): self.save()
        #self.save(timestamp=sec)
                
        return isEvent

    def reset(self):
        for i,s in self.salientMap.iteritems(): 
            s.reset()

    def getInOut(self,exp,i,j):
        return self.salientMap[i].getInOut(exp,self.salientMap[j])

    def inverse(self,i,tin,tout):
        self.salientMap[i].inverse(0,tin,tout)

    def setModelsMode(self,value="explore"):
        for i,s in self.salientMap.iteritems(): 
            models = s.models
            for i,m in enumerate(models):
                models[i].mode = value

    def printModels(self,label,models,in_mins,in_maxs,out_mins,out_maxs):
        print label, "added models: ", len(models)
        for i,m in enumerate(models):
            in_mins[i] = [round(x,1) for x in in_mins[i]]
            in_maxs[i] = [round(x,1) for x in in_maxs[i]]
            out_mins[i] = [round(x,1) for x in out_mins[i]]
            out_maxs[i] = [round(x,1) for x in out_maxs[i]]
            print " ", i,":  ", in_mins[i],"..", in_maxs[i],"  --->  ",out_mins[i],"..",out_maxs[i]
        print ""
        self.setModelsMode('exploit')

    def save(self,salient=[],timestamp=-1):
        folder=self.exp.name + "/"
        for i,s in self.salientMap.iteritems():
            if(len(salient)>0): 
                if(s not in salient): continue
            if(timestamp<0): fname = "data/%ssalient/map%d.npy"%(folder,i)
            else: fname = "data/%ssalient/timestamp/map%d-%d.npy"%(folder,i,timestamp)

            np.save(fname, self.salientMap[i].mat.m)
            for j in range(len(s.models)):
                if(s.models[j].size() > 10):
                    if(timestamp<0): fname = "data/%ssalient/s%d-model%d.data"%(folder,i,j)
                    else: fname = "data/%ssalient/timestamp/s%d-model%d-%d.data"%(folder,i,j,timestamp)
                    pickle.dump(s.models[j], open(fname, "wb"))

    def load_history(self,i=0):
        folder=self.exp.name + "/"
        fname = "data/%shistory%d.npy"%(folder,i)
        return np.load(fname)

    def load_models(self):
        folder=self.folder
        bDebug = True
        if(bDebug): print "Loading Salient Point Models", len(self.salientMap)
        for i,s in self.salientMap.iteritems():
            if(s.type in ['left','right']):
                for j in range(len(s.models)):
                    fname = "data/%s/salient/s%d-model%d.data"%(folder,i,j)
                    if(os.path.isfile(fname)):
                        if(bDebug): print fname, 
                        s.models[j] = pickle.load(open(fname, "rb"))
                        if(bDebug): print "with",s.models[j].size(),"tuples"
            
            if(s.type == 'obj'):
                for j in range(len(s.models)):
                    fname = "data/%s/salient/s%d-model%d.data"%(folder,i,j)
                    if(os.path.isfile(fname)):
                        if(bDebug): print fname, 
                        s.models[j] = pickle.load(open(fname, "rb"))
                        if(bDebug): print "with",s.models[j].size(),"tuples"

    def load(self,i=-1):
        #bDebug = Box2DWorld.bDebug
        bDebug = True
        folder=self.folder
        if(i<0):
            for i in range(len(self.salientMap)):
                self.salientMap[i] = Salient(i,self.exp,self.exp.getSalientType(i))
                fname  = "data/%s/salient/map%d.npy"%(folder,i)
                if(os.path.isfile(fname)):
                    if(bDebug): print "load",fname
                    self.salientMap[i].mat.m = np.load(fname)
                else:
                    if(bDebug): print "Not loaded",fname                    
        else:
            self.salientMap[i].mat.m = np.load("data/%s/salient/map%d.npy"%(folder,i))
            return self.salientMap[i].mat.m

        self.load_models()

