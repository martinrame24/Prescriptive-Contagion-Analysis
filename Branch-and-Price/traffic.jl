# MASTER NOTE: ONLY NEED TO SPECIFY VARIABLES THAT ARE ALL UPPER CASE + COST_FUNCTION + ODE_EQUATION

# ODE simulation time
TSPAN = (0.0, 10.0)
# Time Length of Interventions
#T = 6
# Total number of plans
TOT_PLANS = 500000
# number of regions
N_REGION = 5

# Region Weights
WEIGHT = [1.00 for i in  1:N_REGION]


# Intervention values and budget
#ve = 8
#be = 4
TREATMENT_VALS = [(x,y) for x in 0:ve, y in 0:be]
TREATMENT_BUDGET = [(ve, be) for i in 1:T]

BRANCHING_RANGE = [[[[0, ve], [0,be]] for t in 1:T] for n in 1:N_REGION]

# ODE cost function
wt = 2
function cost_function(sol, region, t)
    cost = wt*(sol[2] + sol[4]) + sol[5]
    return cost
end

function current_cost(region, states)
    return wt*(states[region][2] + states[region][4]) + states[region][5]
end

# Initial state

initial_state = [[0.85, 0, 0.08, 0, 0.07, 0], [0.92, 0, 0.05, 0, 0.03, 0], 
                [0.77, 0, 0.01, 0, 0.22, 0], [0.83, 0, 0.07, 0, 0.1, 0], [0.76, 0, 0.09, 0, 0.15, 0]]

# Definition of parameters for ODE and ODE

alpha_F = [0.0641, 0, 0.0807, 0.0446, 0.5076]
beta_F = [0.0989, 0.0454, 0.1294, 0.0411, 0.04715]
rho_F = [0.3053, 0, 0.1832, 0.3498, 0.551]
mu_F = [0.0640, 0.07394, 0.0524, 0.0878, 0.0613]

alpha_W = [3.034, 1.2536, 3.082, 1.139, 1.393]
beta_W = [0.3390, 0.8634, 2.003, 2.053, 0.9927]
rho_W = [0.1117, 0.5682, 0.2890, 0.3842, 0.44]
mu_W = [2.223, 1.355, 2.873, 1.823, 2.004]

theta =[0.0769, 0.081, 0.0123, 0.0442, 0.0763]

function g(y, mu)
    return mu*(y-be/N_REGION)/2
end

function f(x, mu)
    return mu*(x-ve/N_REGION)/4
end

function ode_equation!(ds, s, p, t)
    
    alpha_rf, beta_rf, rho_rf, mu_rf = alpha_F[p[1]], beta_F[p[1]], rho_F[p[1]], mu_F[p[1]]
    alpha_rw, beta_rw, rho_rw, mu_rw = alpha_W[p[1]], beta_W[p[1]], rho_W[p[1]], mu_W[p[1]]
    
    theta_r = theta[p[1]]
    
    ds[1] = - alpha_rf*s[1]*(s[2] + s[4] + s[5]) - beta_rf*s[1] + rho_rf*(mu_rf + g(p[4], mu_rf))*s[2]
    
    ds[2] = beta_rf*s[1] - rho_rf*(mu_rf + g(p[4], mu_rf))*s[2] - (1-rho_rf)*(mu_rf + f(p[3], mu_rf))*s[2]
    
    ds[3] = - alpha_rw*s[3]*(s[2] + s[4] + s[5]) - beta_rw*s[3] + rho_rw*(mu_rw + g(p[4], mu_rw))*s[4]
    
    ds[4] = beta_rw*s[3] - rho_rw*(mu_rw + g(p[4], mu_rw))*s[4] - (1-rho_rw)*(mu_rw + f(p[3], mu_rw))*s[4]
    
    ds[5] = - theta_r*s[5] + alpha_rf*s[1]*(s[2] + s[4] + s[5]) + alpha_rw*s[3]*(s[2] + s[4] + s[5]) + 
            (1-rho_rf)*(mu_rf + f(p[3], mu_rf))*s[2] + (1-rho_rw)*(mu_rw + f(p[3], mu_rw))*s[4]
    
    ds[6] = theta_r*s[5]
    
end



### Stuff you usually wont touch, just copy over to new application
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
            cost_i = cost_i + cost
        end
        cost_master = cost_master + cost_i
    end
    return cost_master
end