import sys                                # Python includes
import os
import numpy as np
import random
import pickle
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('text', usetex = False)      
sys.path.append('../_utils/')             # Custom includes
sys.path.append('../BimanualRobotSimulation/')
import Box2DWorld                         # Box2D Physics library world
import VectorFigUtils                     
import SensoriMotor
import RobotLearning
import AffordanceGradients as AG
import ExplautoUtils

from explauto import SensorimotorModel    # Explauto library includes
from explauto import InterestModel
from explauto.environment.context_environment import ContextEnvironment
from explauto.interest_model.discrete_progress import DiscretizedProgress, competence_dist
