TSPAN = (0.0, 1)
TOT_PLANS = 500000

N_REGION = 20
#T = 50

### Budget for x is one (it is a proportion) and for y it is #products
K_x = 1
#K_y = 4

### Range of values for x and y (y is always 1)
#D_x = 10
D_y = 1

#Either we do not select the product then x=y=0, or we select it and y=1, 0<x<=D_x
TREATMENT_VALS = vcat([(0,0)], [(x,1) for x in 0:D_x])
#x must sum up to 1 and y to number of products
TREATMENT_BUDGET = [(D_x, K_y) for i in 1:T]
#Only two branchings for y and up to D_x for x
BRANCHING_RANGE = [[[[0, D_x], [0, D_y]] for t in 1:T] for i in 1:N_REGION]
MAX_SHARE = 1

MARKET_SIZE = 1.0
WEIGHT = [1.00 for i in  1:N_REGION]

function cost_function(sol, region, t)
    if t == T
        cost = sol[2]
    else
        cost = 0
    end
    #We return -cost because we want to keep a minimizing structure in the column generation
    return -cost
end

function current_cost(region, states)
    return states[region][2]
end

alpha = [0.143, 0.166, 0.14, 0.155, 0.118, 0.145, 0.123, 0.192, 0.149, 0.163, 0.176, 0.101, 0.163, 0.181, 0.151, 0.16, 0.155, 0.178, 0.112, 0.175, 0.155, 0.131, 0.168, 0.107, 0.134, 0.145, 0.158, 0.143, 0.115, 0.16, 0.097, 0.144, 0.104, 0.183, 0.141, 0.121, 0.143, 0.097, 0.107, 0.176, 0.103, 0.181, 0.171, 0.132, 0.148, 0.139, 0.121, 0.169, 0.098, 0.167, 0.153]
beta = [0.072, 0.075, 0.09, 0.059, 0.067, 0.065, 0.091, 0.083, 0.066, 0.061, 0.068, 0.089, 0.071, 0.055, 0.07, 0.09, 0.059, 0.07, 0.056, 0.088, 0.061, 0.056, 0.083, 0.092, 0.088, 0.086, 0.056, 0.087, 0.079, 0.081, 0.088, 0.087, 0.078, 0.078, 0.084, 0.065, 0.068, 0.091, 0.076, 0.063, 0.088, 0.084, 0.071, 0.071, 0.074, 0.083, 0.088, 0.062, 0.081, 0.089, 0.07]
initial_state = [[MARKET_SIZE*0.99, MARKET_SIZE*0.01] for i in 1:N_REGION]

### Make sure that we use this ode equation
function ode_equation!(ds, s, p, t)
    alpha_r = alpha[p[1]]
    beta_r = beta[p[1]]
    ds[1] = -alpha_r*p[3]*MAX_SHARE*(MARKET_SIZE - s[2])/D_x - beta_r*(MARKET_SIZE - s[2])*s[2]/MARKET_SIZE
    ds[2] = alpha_r*p[3]*MAX_SHARE*(MARKET_SIZE - s[2])/D_x + beta_r*(MARKET_SIZE - s[2])*s[2]/MARKET_SIZE
end

### Stuff you usually won't touch, just copy over to new application
N_DECISION = length(TREATMENT_VALS[1])
DECISION_PT = [[Tuple(repeat([1], N_DECISION)) for t in 1:T] for p in 1:TOT_PLANS]


# Cost of plans
COST_P = [1.0 for p in 1:TOT_PLANS]
# TODO: harmonize indicators
PLAN_REGION_IND = [0 for p in 1:TOT_PLANS]
PLAN_INFEAS_IND = [true for p in 1:TOT_PLANS]
GOOD_BRANCHINGS = true

GLOBAL_BRANCHING_RANGE = deepcopy(BRANCHING_RANGE)
GLOBAL_COST_P = deepcopy(COST_P)
GLOBAL_DECISION_PT = deepcopy(DECISION_PT)
GLOBAL_PLAN_REGION_IND = deepcopy(PLAN_REGION_IND)
GLOBAL_PLAN_INFEAS_IND = deepcopy(PLAN_INFEAS_IND)
GLOBAL_GOOD_BRANCHINGS = deepcopy(GOOD_BRANCHINGS)

function cost_evaluation(plan)
    cost_master = 0
    for i in 1:N_REGION
        initial_state_region = initial_state[i]
        cost_i = 0
        for t in 1:T
            initial_state_region, cost = solve_ode_exact(initial_state_region, plan[i][t], i, t)
            cost_i += cost
        end
        cost_master += cost_i
    end
    return cost_master
end
