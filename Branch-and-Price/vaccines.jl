using CSV, DataFrames, Dates, JuMP, Gurobi
# MASTER NOTE: ONLY NEED TO SPECIFY VARIABLES THAT ARE ALL UPPER CASE + COST_FUNCTION + ODE_EQUATION

# ODE simulation time
TSPAN = (0.0, 7.0)
# Time Length of Interventions
#T = 6
# Total number of plans
TOT_PLANS = 500000
# number of regions
# included_regions = ["I"]
included_regions = ["A","B","C","D","E","F","G","H","I","J"]
regions_to_state_dict = Dict("A" => ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont", "New York"],
                            "B" => ["Delaware", "District of Columbia", "Maryland", "Pennsylvania", "Virginia", "West Virginia", "New Jersey"],
                            "C" => ["North Carolina", "South Carolina", "Georgia", "Florida"],
                            "D" => ["Kentucky", "Tennessee", "Alabama", "Mississippi"],
                            "E" => ["Illinois", "Indiana", "Michigan", "Minnesota", "Ohio", "Wisconsin"],
                            "F" => ["Arkansas", "Louisiana", "New Mexico", "Oklahoma", "Texas"],
                            "G" => ["Iowa", "Kansas", "Missouri", "Nebraska"],
                            "H" => ["Colorado", "Montana", "North Dakota", "South Dakota", "Utah", "Wyoming"],
                            "I" => ["Arizona", "California", "Hawaii", "Nevada"],
                            "J" => ["Alaska", "Idaho", "Oregon", "Washington"])

regions = [a for x in included_regions for a in regions_to_state_dict[x]]
N_REGION = sum([length(regions_to_state_dict[x]) for x in included_regions])
regions_mapping_dict = Dict(zip(collect(1:N_REGION), regions))
# Region Weights
WEIGHT = [1.00 for i in  1:N_REGION]

#Intervention values and budget
vaccine_budget = 50000 * N_REGION
#num_treatment_vals = 10
TREATMENT_VALS = [(x,) for x in 0:num_treatment_vals]
TREATMENT_BUDGET = [(num_treatment_vals, ) for i in 1:T]
BRANCHING_RANGE = [[[[0, num_treatment_vals]] for t in 1:T] for n in 1:N_REGION]
vaccine_unit = vaccine_budget / num_treatment_vals

# ODE cost function
lambda_cases = 0
# NEED TO SPECIFY
function cost_function(sol, region, t)
    if t == T
        cost = (sol[8] + sol[9] + sol[11] + lambda_cases * sol[2]) * region_population_dict[regions_mapping_dict[region]]
    else
        cost = 0
    end
    return cost
end

# Initial state
delphi_params = CSV.read("data/vaccine/delphi-parameters.csv", DataFrame)
delphi_params_us = delphi_params[delphi_params.Country .== "US", :]
region_parameters_dict = Dict(zip(delphi_params_us.Province, collect(eachrow(Matrix(delphi_params_us[:,6:end])))))
delphi_params = CSV.read("data/vaccine/delphi-parameters.csv", DataFrame)

population = CSV.read("data/vaccine/population.csv", DataFrame)

population_combined = combine(DataFrames.groupby(population, :state),:population => sum)
region_population_dict = Dict(zip(population_combined.state, population_combined.population_sum))

delphi_predictions = CSV.read("data/vaccine/delphi-predictions.csv", DataFrame)
delphi_predictions_us = delphi_predictions[(delphi_predictions.Country .== "US") .& (delphi_predictions.Day .== Dates.Date("2021-02-01")), :]
initial_state = [vcat(Array(delphi_predictions_us[delphi_predictions_us.Province .== region,:][1,5:15]),[0,0,0,0]) for region in regions]
initial_state = [x ./ sum(x) for x in initial_state]

IncubeD = 5
RecoverID = 10
RecoverHD = 15
DetectD = 2
VentilatedD = 10  # Recovery Time when Ventilated
p_v = 0.25  # Percentage of ventilated
p_d = 0.2  # Percentage of infection cases detected.
p_h = 0.03  # Percentage of detected cases hospitalized
vac_effect = 0.85

# NEED TO SPECIFY
function ode_equation!(ds, s, p, t)
    """
    SEIR based model with 16 distinct states, taking into account undetected, deaths, hospitalized and
    recovered, and using an ArcTan government response curve, corrected with a Gaussian jump in case of
    a resurgence in cases
    """
    region = regions_mapping_dict[p[1]]
    t_eff = t + (p[2] - 1) * TSPAN[2]
    alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3 = region_parameters_dict[region]
    N = region_population_dict[region]
    r_i = log(2) / IncubeD  # Rate of infection leaving incubation phase
    r_d = log(2) / DetectD  # Rate of detection
    r_ri = log(2) / RecoverID  # Rate of recovery not under infection
    r_rh = log(2) / RecoverHD  # Rate of recovery under hospitalization
    r_rv = log(2) / VentilatedD  # Rate of recovery under ventilation
    gamma_t = (
        (2 / pi) * atan(-(t_eff - days) / 20 * r_s) + 1
        + jump * exp(-(t_eff - t_jump) ^ 2 / (2 * std_normal ^ 2))
    )
    p_dth_mod = (2 / pi) * (p_dth - 0.001) * (atan(-t_eff / 20 * r_dthdecay) + pi / 2) + 0.001
    # Equations on main variables
    Vt = p[3] * vaccine_unit / N
    Vt = min(s[1] / vac_effect, Vt)
    ds[1] = -alpha * gamma_t * (s[1] - vac_effect * Vt) * (s[14] + s[3]) - vac_effect * Vt
    ds[2] = alpha * gamma_t * (s[1] - vac_effect * Vt) * (s[14] + s[3]) - r_i * s[2]
    ds[3] = r_i * s[2] - r_d * s[3]
    ds[4] = r_d * (1 - p_dth_mod) * (1 - p_d) * s[3] - r_ri * s[4]
    ds[5] = r_d * (1 - p_dth_mod) * p_d * p_h * s[3] - r_rh * s[5]
    ds[6] = r_d * (1 - p_dth_mod) * p_d * (1 - p_h) * s[3] - r_ri * s[6]
    ds[7] = r_d * p_dth_mod * (1 - p_d) * s[3] - r_dth * s[7]
    ds[8] = r_d * p_dth_mod * p_d * p_h * s[3] - r_dth * s[8]
    ds[9] = r_d * p_dth_mod * p_d * (1 - p_h) * s[3] - r_dth * s[9]
    ds[10] = r_ri * (s[4] + s[6]) + r_rh * s[5]
    ds[11] = r_dth * (s[7] + s[8] + s[9])
    # vaccine states
    ds[12] = - alpha * gamma_t * (s[12] + vac_effect * Vt) * (s[14] + s[3]) + vac_effect * Vt
    ds[13] = alpha * gamma_t * (s[12] + vac_effect * Vt) * (s[14] + s[3]) - r_i * s[13]
    ds[14] = r_i * s[13] - r_d * s[14]
    ds[15] = r_d * s[14]
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