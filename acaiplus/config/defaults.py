from yacs.config import CfgNode

_C = CfgNode()

###############
# DIRECTORIES #
###############
_C.DIRS = CfgNode()
_C.DIRS.OUTPUT = "/output/folder/path/"
_C.DIRS.DATA = "/path/to/save/datasets/"

##########
# Models #
##########
_C.MODEL = CfgNode()
_C.MODEL.ARCH = ""
_C.MODEL.DEVICE = "cuda"
_C.MODEL.MODE = "train"  # train or cluster
_C.MODEL.SAVE_INTERVAL = 100
_C.MODEL.ADVERSARY = False
_C.MODEL.ADDED_CONSTR = True

# general model hyperparams
_C.MODEL.DEPTH = 16
_C.MODEL.LATENT = 16
_C.MODEL.LATENT_WIDTH = 4
_C.MODEL.NEG_SLOPE = 0.2
_C.MODEL.REG = 0.2
_C.MODEL.ADVWEIGHT = 0.5
_C.MODEL.ADVDEPTH = 0

# IDEC Models
_C.MODEL.IDEC = CfgNode()
_C.MODEL.IDEC.GAMMA = 0.1
_C.MODEL.IDEC.PRETRAIN = 300


##########
# Solver #
##########
_C.SOLVER = CfgNode()
_C.SOLVER.EPOCHS = 400
_C.SOLVER.LR = 0.0001
_C.SOLVER.SEED = 51
# type epoch nr or "max" for latest or "0" to not load models.
_C.SOLVER.RESUME_EPOCH = "max"


########
# data #
########
_C.DATA = CfgNode()
_C.DATA.DATASET = "mnist"
_C.DATA.BATCH_SIZE = 64
_C.DATA.COLORS = 1
_C.DATA.WIDTH = 32
_C.DATA.HEIGHT = 32
_C.DATA.NCLASS = 10
_C.DATA.TEMPORAL_BOOL = False


################
# Visualiation #
################
_C.VIZ = CfgNode()
_C.VIZ.NR_LINES = 16
_C.VIZ.GRID_LINES = 16
_C.VIZ.PLOT_DATA_INDEX = 1
_C.VIZ.SAVE_SVD_PLOTS = False
_C.VIZ.SVD_COMPONENTS = 40
_C.VIZ.LATENT_DIGIT = 5


##############
# Clustering #
##############
_C.CLUSTER = CfgNode()
_C.CLUSTER.N_RUNS = 1000

#####################
# Temporary Results #
#####################
_C.RESULTS = CfgNode()
