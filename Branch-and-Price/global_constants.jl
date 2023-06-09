#global T_VALUES = [4,6,8,10,12]
#global T_VALUES = [4]
#global D_VALUES = [2,5,10,20,40]
#global D_VALUES = [4,2]
#global TD_VALUES = [[T_VALUES[i], D_VALUES[j]] for i in 1:length(T_VALUES) for j in 1:length(D_VALUES)]
#global TD_VALUES = [[2,2]]

global PRECISION_DIGITS = 20
#global BATCHED_NORM_BOUND = 0.01  initial value for clustering
#global BATCHED_NORM_BOUND = 0.01

global ALREADY_OPTIMIZED = false
global NEGATIVE_COLUMN = false

global C_BUDGET = []
global C_REGION = []
global C_CONSISTENT = []
global F_BUDGET = []
global F_CAPACITY = []
global BUDGET = []

# Helper Constants for print statements
global RECORD_NOW = 1
global RECORD_END = 1


global EPSILON = 1e-5
global MAX_VAL = 1e6
# Branch and Price constants
global NCOLB = 100000

global LB_NODE = [0.0 for n in 1:NCOLB]
global PRE_NODE = [1 for n in 1:NCOLB]
global SIGN_NODE = [true for n in 1:NCOLB]
global THRE_NODE = [[1, 1, 1.0, 0] for n in 1:NCOLB]
global CUR_NODE = 1
global TOT_NODE = 0
global UB = Inf
global LB = -Inf    