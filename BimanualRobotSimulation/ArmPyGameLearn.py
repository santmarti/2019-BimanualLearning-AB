import time
import pygame
from pygame.locals import *
import pygame.surfarray as surfarray

import numpy as np
from numpy import linalg as npla
import random
import pickle
import sys
sys.path.append('../_utils/')
import PyGameUtils
import Box2DWorld
import RobotLearning
import SensoriMotor
import VectorFigUtils

box2dWH = (PyGameUtils.SCREEN_WIDTH, PyGameUtils.SCREEN_HEIGHT)

#***************************
#PYGAME initialization
#***************************
pygame.init()
screen = pygame.display.set_mode(box2dWH, 0, 32)
surfarray.use_arraytype('numpy')
pygame.display.set_caption('Arm Simulation Learning')
clock=pygame.time.Clock()

circles = []
isalient = 0  # salient map to draw: select with 1,2,3...

#target_pos = [0,2]
target_angle = 0

#fontname = pygame.font.get_default_font()
#font = pygame.font.SysFont(fontname, 25, True, False)

#X0 = 5 * Box2DWorld.PPM
#Y0 = 1.7 * Box2DWorld.PPM
exp = Box2DWorld.ExpSetup(obj_type="box", salientMode = "minimum", debug = True, name ="human",bSelfCollisions=True)
nao = exp.nao
obj = exp.obj

#if(i==0):   x_lim, ylim = (130,400), (50,250)
#elif(i==2): x_lim, ylim = (190,450), (50,250)


def experiments(learnModel):
    global exp
    exp.setObjPos([0,4])

    # Resultados 7 horas
    #n = 1000
    #tests_mdm=[[n,1,0,0],[n,5,0,0],[n,10,0,0],[1,1,n,1],[1,1,n,5],[1,1,n,10]] 
    #tests_m=[[10*n,0],[1,10*n]]

    n = 2500
    tests_mdm=[[1,1,n/5,5]] 
    tests_m=[[0,n]]

    for e in tests_mdm:
        nm,ndm,nb,nbdm = e
        strid = "dm-%d-%d-%d-%d"%(nm,ndm,nb,nbdm)
        learnModel.iniExp("twoarms",strid)
        print "Starting f(mdm)=sds",strid,"-----------------------------------------"
        learnModel.learnDeltaArms(strid,nmotor=nm,ndmmotor=ndm,nbabbling=nb,ndmbabbling=nbdm)     # f(m,dm)=s,s learn from dm 

    for e in tests_m:
        nm,nb = e
        strid = "m-%d-%d"%(nm,nb)
        learnModel.iniExp("twoarms",strid)
        print "Starting f(m)=s",strid,"-----------------------------------------"
        learnModel.learnNoDeltaArms(strid,nmotor=nm,nbabbling=nb)     # f(m,dm)=s,s learn from dm 


learnModel = RobotLearning.LearningModel(exp)

learnModel.load(folder='biobj',strid='arm')
#experiments(learnModel)

#learnModel.iniExp("human","arm")
#learnModel.load('arm')

learnModel.load('arm')
#learnModel.learnNoDeltaArms()     # f(m)=s,s two arms 



def draw_velocity(screen, s):
    p = np.array(s.getxy())
    v = np.array(s.getv())
    q = p + v
    PyGameUtils.my_draw_line(screen, [p,q], color = (10,80,250,10))

def draw_velocities(screen, SM):
    sobj = SM.getSalientObj()
    sleft = SM.getSalientLeft()
    sright = SM.getSalientRight()
    draw_velocity(screen,sobj)
    draw_velocity(screen,sleft)
    draw_velocity(screen,sright)


def draw_targets(learnModel,iarm = -1):
    global screen, nao, v_g
    if(len(v_g) >= 2):
        p = learnModel.SM.getSalientObj().getxy()
        q = np.array(p) + 1.5*np.array(v_g[:2])
        PyGameUtils.my_draw_line(screen, [p,q], color = (10,170,250,255))

    # list_s = [learnModel.targets_sds(imodel=3,mode='mat',doit=False) for _ in range(20)]            
    # for s in list_s:
    #     PyGameUtils.box2d_draw_mycircle(screen, s[:2], radius = 0.08, color=(27,200,7))
    #     PyGameUtils.box2d_draw_mycircle(screen, s[2:], radius = 0.08, color=(27,200,7))

    arms = [0,1]
    if(iarm >= 0): arms = [iarm]
    if(not nao.bTwoArms): arms = [0]

    for iarm in arms:
        if(len(learnModel.goals[iarm]) > 0):
            position = learnModel.goals[iarm]
            #PyGameUtils.box2d_draw_mycircle(screen, position, radius = 0.08, color=(27,200,7))


def arrayShow(array_img):
    global screen,box2dWH
    "displays a surface, waits for user to continue"
    mat = array_img
    if(mat.shape == box2dWH):
        mat = np.dstack((mat,mat,mat))  # add RGB components

    surfarray.blit_array(screen, mat)

def event_handle():
    global nao,running,isalient
    d = 1
    dm = np.array([d,d,d])
    for event in pygame.event.get(): # Check the event queue
        if(event.type!=KEYDOWN): continue

        if(event.key== pygame.K_LEFT): nao.arms[0].deltaMotor(dm)
        if(event.key== pygame.K_RIGHT): nao.arms[0].deltaMotor(-dm)
        if(event.key== pygame.K_UP): nao.arms[1].deltaMotor(dm)
        if(event.key== pygame.K_DOWN): nao.arms[1].deltaMotor(-dm)

        if(event.key== pygame.K_SPACE): learnModel.SM.setObjPos()
        if(event.key== pygame.K_d): 
            print nao.arms[0].deltaMotor()

        if(event.key== pygame.K_s): learnModel.SM.save()

        if(event.key== pygame.K_RETURN): 
            #learnModel.learnDeltaArms()
            s = learnModel.targets_sds(imodel=3, mode='mat', doit=True, online=False)
            s_real = nao.getFinalPos() 
            print "mse:", npla.norm(s-s_real)**2
            # small errors because generated with targets_sds in reach




        if(event.key== pygame.K_1): isalient = 0
        if(event.key== pygame.K_2): isalient = 1
        if(event.key== pygame.K_3): isalient = 2
        if(event.key== pygame.K_4): isalient = 3
        if(event.key== pygame.K_5): isalient = 4
        if(event.key== pygame.K_6): isalient = 5
        if(event.key== pygame.K_7): isalient = 6
        if(event.key== pygame.K_8): isalient = 7
        if(event.key== pygame.K_9): isalient = 8


        if event.type==QUIT or event.key==K_ESCAPE:
            # The user closed the window or pressed escape
            running=False


s = screen.get_size()
img_size = np.zeros((screen.get_size()))


niter = 0
iarm = 1
itarget = 0
itrials = 0
targets = []
running=True


# learn Bimanual variables
v_g = []
sobj_pos = []
sobj_angle = 0
#bEvaluation = False
bEvaluation = True
dirs = [ [round(np.cos(a)/5.0,2), round(np.sin(a)/5.0,2),0] for a in [i*np.pi/8.0 for i in range(2*8)]]
#dirs = [[0,0.1,0]]
ErrorsBimanual = []

def learnBimanual():
    global niter, itarget, itrials, iarm, exp, dm, v_g, sobj_pos, circles, bEvaluation
    if(exp.salientMode != "laterals"): print exp.salientMode, "should be laterals"

    err = exp.update()
    sobj = learnModel.SM.getSalientObj()
    sobj_pos = sobj.getxy()
    hleft = exp.getFinalHaptic(arm=0)
    hright = exp.getFinalHaptic(arm=1)

    c_dims = [2,3]
    im = sobj.index_bimanual
    niter += 1
    if(itarget==0): # when target reached
        niter = 0
        exp.setObjPos([0,4])
        nao.restPosition()
        err = exp.update()
        if(err < .02): 
            d = 1.2
            #learnModel.SM.save([sobj])
            obj_pos = exp.ini_obj_pos + d*np.random.rand(2) - d/2.0
            exp.setObjPos(obj_pos)
            niter, itarget = 0, 1
    elif(itarget==1): # when target reached
        learnModel.setGrabObjGoal()    
        if(niter > 300 or (niter > 50 and (hleft > 0.1 or hright > 0.1))):
            v_g = sobj.im_models[im].sample_given_context(sobj_pos, c_dims)
            if(sobj.models[im].size()>0): sdm = sobj.models[im].inverse_prediction(np.hstack((v_g,sobj_pos)))
            else: sdm = sobj.env[im].random_motors(n=1)['0']
            sdm = RobotLearning.min_max(sdm, sobj.in_mins[im], sobj.in_maxs[im])
            s,dm = sdm[0:2], sdm[2:]
            print "IM v_g:", v_g, "dm", dm,
            sys.stdout.flush()
            exp.deltaMotor(dm)
            niter,itarget = 0,2

    elif(itarget==2): # when target reached
        v_s = sobj.getv()
        if(niter > 20):
            print "   stored", v_s
            if(hleft > 0.5 or hright > 0.5):
                sobj.models[im].update(np.hstack((sobj_pos, dm)), np.hstack((sobj_pos, v_s)))
                #sobj.im_model[im].update(np.hstack((sobj_pos, dm, v_g)), np.hstack((sobj_pos, v_s)))
            niter,itarget,itrials = 0,1,itrials + 1
            if(itrials > 10): itrials, itarget = 0,0


def learnBimanualSensory():
    global  learnModel, niter, itarget, itrials, iarm, exp, circles, v_g, sobj_pos, sobj_angle, bEvaluation, dirs, ErrorsBimanual
    SM = learnModel.SM

    err = exp.update()
    sobj = SM.getSalientObj()
    imodel = sobj.index_bimanual
    sobj_pos = sobj.getxy()

    si = SM.getSalientObj(dir=-1)
    sj = SM.getSalientObj(dir=1)
    hleft = exp.getFinalHaptic(arm=0)
    hright = exp.getFinalHaptic(arm=1)

    niter += 1
    if(itarget==0): # when target reached
        exp.setObjPos([0,4])
        nao.restPosition()
        err = exp.update()
        if(err < .02): 
            learnModel.setObjTargetPos(d=0.1,bTarget=False)
            niter, itarget = -150, 1

    elif(itarget==1): # when target reached
        learnModel.setGrabObjGoal(ini_far=0.35) 
        if(niter > 90 or (niter > 10 and hleft >= 0.9 and hright >= 0.9)):
            if(imodel == 6): c_dims = [3,4]
            elif(imodel == 7): c_dims = [2,3]

            if(not bEvaluation): 
                sobj.models[imodel].mode = 'explore'
                v_g = sobj.im_models[imodel].sample_given_context(sobj_pos, c_dims)
            else:   
                sobj.models[imodel].mode = 'exploit'
                v_g = random.choice(dirs)

            if(sobj.models[imodel].size()>0): tin = sobj.models[imodel].inverse_prediction(np.hstack((v_g,sobj_pos)))
            else: tin = sobj.env[imodel].random_motors(n=1)['0']
            
            tin = RobotLearning.min_max(tin, sobj.in_mins[imodel], sobj.in_maxs[imodel])
            vi,vj= 3*np.array(tin[0:2]), 3*np.array(tin[2:4])
            learnModel.gotoDeltaS(si,vi,iarm=0)
            learnModel.gotoDeltaS(sj,vj,iarm=1)

            sobj_angle = exp.obj.angle
            print "Evaluation: ", bEvaluation, "angle",VectorFigUtils.vangleSign(v_g[:2],[1,0]),"     v_g:", v_g,
            sys.stdout.flush()

            niter,itarget = 0,2

    elif(itarget==2): # when target reached
        vi, vj = si.getv(), sj.getv()
        vobj = sobj.getv()
        #if(sobj.getvnorm() > 0.01): vobj = sobj.getvnormalized()
        #else: vobj = [0,0]            
        angle_diff = exp.obj.angle - sobj_angle 

        if(niter > 18):
            if(si.getvnorm() >= 0.001 and si.getvnorm() >= 0.001 and npla.norm(vobj) >= 0.001):         
                err_angle = VectorFigUtils.vangle(v_g[:2],vobj)
            else: 
                err_angle = 2*np.pi
            
            err_goal = npla.norm(np.array(v_g[:2]) - np.array(vobj))
            
            if(bEvaluation):
                if(hleft > 0.2 and hright > 0.2 and err_angle < 2*np.pi):
                    ErrorsBimanual.append([err_angle,err_goal,v_g[0],v_g[1]])
                    print "err: ", round(err_goal,2), "err_angle", round(err_angle,2) 
                    if(len(ErrorsBimanual) % 5 == 0):
                        fname = "data/bimanual_error.data"
                        pickle.dump(ErrorsBimanual, open(fname, "wb"))
                        #bEvaluation = not bEvaluation
                        print "EVAL PHASE:",  np.array(ErrorsBimanual)[-5:,0]
            #elif(not bEvaluation and hleft > 0.1 and hright > 0.1):
            elif(not bEvaluation):
                print "      stored", vobj, " angle", round(angle_diff,2), "err: ", round(err_goal,2), "err_angle", round(err_angle,2) 
                tin = np.hstack((vi, vj))                
                if(imodel == 6):
                    tout = np.hstack((vobj,angle_diff,sobj_pos))
                    tout_goal = np.hstack((v_g,sobj_pos))
                elif(imodel == 7):
                    tout = np.hstack((vobj,sobj_pos))
                    tout_goal = np.hstack((v_g,sobj_pos))
                sobj.errorManager.goalBabblingUpdate(tin,tout,tout_goal,imodel)
                #if(sobj.models[imodel].size()%10==0): 
                    #bEvaluation = not bEvaluation
                    #SM.save()
            elif(not bEvaluation):
                print "  NOT STORED", hleft, hright, "sobj vnorm:", sobj.getvnorm()

            for _ in range(12): 
                learnModel.setGrabObjGoal(ini_far=0.6) 
                Box2DWorld.step()
                exp.update()

            niter,itarget = 0,1
            itrials += 1
            if(itrials > 10):
                itrials, itarget = 0,0

def reactiveBimanualSensory():
    global  learnModel, niter, itarget, itrials, iarm, exp, dm, v_g, sobj_pos, circles
    SM = learnModel.SM
    err = exp.update()
    sobj = SM.getSalientObj()
    sobj_pos = sobj.getxy()
    hleft = exp.getFinalHaptic(arm=0)
    hright = exp.getFinalHaptic(arm=1)
    c_dims = [2,3]
    im = sobj.index_bimanual
    niter += 1
    if(itarget==0): # when target reached
        niter = 0
        exp.setObjPos([0,4])
        nao.restPosition()
        err = exp.update()
        if(err < .02): 
            learnModel.setObjTargetPos()
            niter, itarget = 0, 1
    elif(itarget==1): # when target reached
        learnModel.setGrabObjGoal(ini_far=0.35) 
        if(niter > 90 or (niter > 10 and hleft >= 0.9 and hright >= 0.9)):
            sl = SM.getSalientObj(dir=-1)
            sr = SM.getSalientObj(dir=1)
            vl = SM.getTargetDir(dir=-1)
            vr = SM.getTargetDir(dir=1)
            learnModel.gotoDeltaS(sl,vl,iarm=0)
            learnModel.gotoDeltaS(sr,vr,iarm=1)
            niter,itarget = 0,2

    elif(itarget==2): # when target reached
        v_s = sobj.getv()
        if(niter > 15):
            for _ in range(12): 
                learnModel.setGrabObjGoal(ini_far=0.6) 
                Box2DWorld.step()
                exp.update()
            niter,itarget = 0,1
            itrials += 1
            if(itrials > 12):
                itrials, itarget = 0,0

def setObjTargetLaterals():
    global niter,iarm,exp,itarget
    if(exp.salientMode != "laterals"): 
        print exp.salientMode, "should be laterals"
    err = exp.update()
    if(itarget==0): # when target reached
        niter, itarget  = 0,1
        learnModel.SM.setObjPos([0,1.5])
        nao.restPosition()
        err = exp.update()
    elif(itarget>0 and err < .05 or niter > 100): # when target reached
        learnModel.setGrabObjGoal()        
        itarget = (itarget+1)%10000
    else:
        niter += 1
    

def setObjTarget():
    global niter,iarm,exp
    err = exp.update()
    err = nao.update(arm=iarm)
    learnModel.setSalientGoal(imodel = iarm, isal=4)
    if(err < .05 or niter > 100): # when target reached
        niter = 0
        iarm = (iarm+1) % 2
        nao.restPosition(otherarm=iarm)
        learnModel.setSensoryGoals(imodel = iarm)
        learnModel.SM.update(exp)
    else:
        niter += 1


def playWithObj(): # random targets arround salient point in the object, alternating hands
    global itarget, niter,iarm,exp,nao
    err = exp.update()
    learnModel.SM.update(exp)
    if(itarget == 0 and err < .05): # when target reached
        niter,iarm,itarget  =  0, (iarm+1) % 2, 1
        nao.restPosition()
        learnModel.iniGoals()
    elif(itarget > 0 and err < .05 or niter > 100):
        if(itarget==1):
            d = 1.3
            obj_pos = exp.ini_obj_pos + d*np.random.rand(2) - d/2.0
            obj_angle = np.random.rand()*2*np.pi - np.pi
            learnModel.SM.setObjPos(obj_pos, obj_angle)

        alfa = np.random.rand()*2*np.pi - np.pi 
        v = np.array(VectorFigUtils.vrotate([1,0],alfa))
        iSalObj = len(nao.getSalient())
        target = np.array( exp.salient[iSalObj] ) + 0.3*v        
        tin,tout = learnModel.SM.getInOut(exp,iSalObj,0)
        niter,itarget  =  0, (itarget+1)%5
        learnModel.gotoS(iarm=iarm, s=target, online=True)
    else:
        niter += 1


def onlineIK():
    global niter,iarm,exp,nao
    err = exp.update()
    if(err < .05 or niter > 100): # when target reached
        niter,iarm  =  0, (iarm+1) % 2
        nao.restPosition(online=True)
        learnModel.setSensoryGoals(iarm = iarm)
        learnModel.SM.update(exp)
    else:
        niter += 1


def onlineDelta():
    global niter,iarm,exp,nao,itarget,learnModel,itrials
    SM = learnModel.SM
    err = exp.update()
    if(iarm == 0): sarm = SM.getSalientLeft() # sarm : point on the arm
    else: sarm = SM.getSalientRight()

    if(itarget==0): # when target reached
        niter,itrials,itarget  = 0,0,1
        nao.restPosition(online=False,otherarm=iarm)
        notFound = True
        while notFound:
            t = sarm.env[1].getRandomInput()
            s = t[:2]
            m = learnModel.models[iarm].inverse_prediction(s)
            s_real = learnModel.gotoS(iarm=iarm, s=s, online=False)
            notFound = VectorFigUtils.dist(s,s_real) > 0.1
        SM.reset()

    elif(itarget==1): # when target reached
        itarget = 2
        t = sarm.env[1].getRandomInput()
        dm = t[2:]
        nao.arms[iarm].deltaMotor(dm)

    elif(itarget>1): # when target reached
        niter += 1
        if(niter > 15): 
            niter,itarget = 0,1
            itrials += 1
            if(itrials == 10): 
                iarm,itarget = (iarm+1)%2,0



def draw_circles():
    global circles
    for c in circles:
        PyGameUtils.box2d_draw_mycircle(screen, c, radius = 0.08, color=(67,100,127))


exp.start()
while running:
    event_handle()
    screen.fill((0,0,0,0))

    learnModel.SM.update(exp)
    exp.update()

    #onlineIK()            # both hands and suitable for obj
    #playWithObj()         # 5 approx to object
    
    #if(int(10*time.time()) % 10==0):  
    #    learnModel.learnDeltaArms() 


    #onlineDelta()

    #setObjTarget()
    #setObjTargetLaterals()

    #learnBimanual()

    learnBimanualSensory()
    #reactiveBimanualSensory()

    #if(learnModel.SM.existsMap(isalient)):
    #    arrayShow(learnModel.SM.getStateMat(isalient))



    PyGameUtils.draw_grid(screen)
    PyGameUtils.draw_world(screen)

    Box2DWorld.step()

    PyGameUtils.draw_contacts(screen, exp)
    PyGameUtils.draw_salient(screen, exp)

    draw_targets(learnModel)

    draw_circles()

    draw_velocities(screen,learnModel.SM)

    pygame.display.flip()              # Flip the screen and try to keep at the target FPS
    clock.tick(Box2DWorld.TARGET_FPS)
    
    pygame.display.set_caption("FPS: {:6.3}{}".format(clock.get_fps(), " "*5))
    
pygame.quit()
print('Done!')


