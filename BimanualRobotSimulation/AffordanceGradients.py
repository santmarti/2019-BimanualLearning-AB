import numpy as np
import random
import pickle
import os.path

import Box2DWorld
import VectorFigUtils
from VectorFigUtils import drawBox, drawCircle
import SensoriMotor
from SensoriMotor import RobotArmEnv

from explauto import Environment
from explauto import SensorimotorModel
from explauto import InterestModel
import ExplautoUtils


def plotDeltaSalient(ax, s, p):  # salient, position
    p = np.array(p)
    v = [1,0]
    a, da = 0, 2*np.pi / 20.0
    while(a < 2*np.pi):
        va = np.array(VectorFigUtils.vrotate(v,a))
        vin = -0.5*va
        tout = np.hstack((vin,p))
        d = ExplautoUtils.distInvNN(s.models[1],tout)
        if(d<0.9):
            vin *= 0.4*(1-d)
            #vi = 0.4*vin
            ax.arrow(p[0], p[1], vin[0], vin[1], head_width=0.04, head_length=0.07, fc='#bbccff', ec='#aaaaff')
        a += da


def plotAG(ax, exp, si, j, pobj):
    v = [1,0]
    a, da = 0, 2*np.pi / 20.0
    pobj = np.array(pobj)
    while(a < 2*np.pi):
        va = np.array(VectorFigUtils.vrotate(v,a))
        #p,pf = pobj + 0.7*va, pobj + 0.7*va
        p,pf = pobj + 0.5*va, pobj + 0.5*va 
        pin = pobj + 0.2*va
        #pin = pobj + [0,0.2]
        vin = -0.8*va
        #vin = [0,0.11]
        tin = np.hstack((pin,vin,pobj))
        fv = si.models[j].forward_prediction(tin)
        d = ExplautoUtils.distFwdNN(si.models[j],tin)
        vel = 2*np.array(fv[0:2])
        #print s4.models[2].infer([4,5,6,7,8,9],[0,1,2,3],t)
        if(d<0.85):
            drawCircle(ax,p,.1*d,color='r')  
            ax.arrow(pf[0], pf[1], vel[0], vel[1], head_width=0.07, head_length=0.11, fc='k', ec='k')
        a += da


def plotAGVectorField(plt,ax,vx,vy,ap,an,shape=[]):
    w,h = ap.shape
    for i in range(w):
        for j in range(h):
            if(i%2==0 and j%2==0):
                dx,dy = vx[i][j], vy[i][j]
                dx,dy = dx/3,dy/3
                if(VectorFigUtils.vnorm([dx,dy])>0.5):
                    ax.arrow(i-w/2, j-h/2, dx, dy, head_width=0.5, head_length=0.5, fc='k', ec='k',label='effect vector field')

    x=np.linspace(0, h, h)
    y=np.linspace(0, w, w)
    x,y=np.meshgrid(x, y)

    if(np.max(ap) > 0.1):
        c1 = plt.contour(x-h/2, y-w/2, ap, colors='g',label="rot plus")

        c2 = plt.contour(x-h/2, y-w/2, an, colors='b',label="rot minus")

    if(len(shape)>0):
        c3 = plt.contour(x-h/2, y-w/2, shape, colors='r',label="shape")


