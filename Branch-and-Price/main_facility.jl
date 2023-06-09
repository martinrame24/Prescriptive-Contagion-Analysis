using JuMP, Gurobi, IterTools, DifferentialEquations, ProgressMeter, LinearAlgebra, PyCall, Revise, DataFrames, CSV, Base

"""
    ode_solution_rounding(L, DIS = 1 / (10 ^ PRECISION_DIGITS))
    rounds the ODE solution to the number of digits while still keeping everything adding up to 1
"""
function ode_solution_rounding(L, DIS = 1 / (10 ^ PRECISION_DIGITS))
    rou_L = L ./ DIS
    res = rou_L .% 1
    residual = round(sum(res), digits = 0)
    rou_L = floor.(rou_L)
    while residual > 0
        max_ind = -1
        max_val = -1
        _ , max_ind = findmax(res)
        rou_L[max_ind] = rou_L[max_ind] + 1
        res[max_ind] = - 1
        residual = residual - 1
    end
    return broadcast(function(x) round(x, digits = PRECISION_DIGITS) end, rou_L .* DIS)
end

"""
    solve_ode(initial_state, decision_vars, region)
    solves the ODE that is defined in the problem specific file
"""
function solve_ode(initial_state, decision_vars, region, t; ode_accuracy, ode_accuracy_level)
    u0 = Array{Float64}(initial_state)
    tspan = TSPAN
    p = vcat([region, t],collect(decision_vars))
    prob = ODEProblem(ode_equation!, u0, tspan, p)
    if ode_accuracy == :auto
        sol = DifferentialEquations.solve(prob)
    elseif ode_accuracy == :fixed
        sol = DifferentialEquations.solve(prob, save_everystep = false, abstol = ode_accuracy_level[1], reltol = ode_accuracy_level[2])
    end
    cost = cost_function(last(sol), region, t)
    sol_states = ode_solution_rounding(last(sol))
    sol_states = max.(sol_states, 0)
    return sol_states, cost
end


function solve_ode_exact(initial_state, decision_vars, region, t)
    u0 = Array{Float64}(initial_state)
    tspan = TSPAN
    p = vcat([region, t],collect(decision_vars))
    prob = ODEProblem(ode_equation!, u0, tspan, p)
    sol = DifferentialEquations.solve(prob)
    cost = cost_function(last(sol), region, t)
    sol_states = max.(last(sol), 0)
    return sol_states, cost
end

"""
    generate_max_bound_state(initial_state, state, region)
    Generate the maximum possible value for a given state and initial condition in a region.
"""
function generate_max_bound_state(initial_state, state, region; ode_accuracy, ode_accuracy_level)
    UB_state = [initial_state[region][state]]
    max_state = [initial_state[region][x] for x in 1:length(initial_state[region])]
    for t in 1:T
        max_s = -0.1
        max_temp = []
        for var_comb in TREATMENT_VALS
            sol_states, cost = solve_ode(max_state, var_comb, region, t, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            cur_state = [x for x in sol_states]
            if cur_state[state] > max_s
                max_s = cur_state[state]
                max_temp = cur_state
            end
        end
        max_state = max_temp
        max_s = round(min(1, max_s + 0.02), digits = PRECISION_DIGITS)
        push!(UB_state, max_s)
    end
    return UB_state
end

"""
    generate_min_bound_state(initial_state, state, region)
    Generate the minimum possible value for a given state and initial condition in a region.
"""
function generate_min_bound_state(initial_state, state, region; ode_accuracy, ode_accuracy_level)
    LB_state = [initial_state[region][state]]
    min_state = [initial_state[region][x] for x in 1:length(initial_state[region])]
    for t in 1:T
        min_s = 1.1
        min_temp = []
        for var_comb in TREATMENT_VALS
            sol_states, cost = solve_ode(min_state, var_comb, region, t, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            cur_state = [x for x in sol_states]
            if minimum(cur_state)<0
                println(min_state)
                println(var_comb)
                println(sol_states)
                println(region)
                println(t)
                return NaN
            end
            if cur_state[state] < min_s
                min_s = cur_state[state]
                min_temp = cur_state
            end
        end
        min_state = min_temp
        min_s = round(max(0, min_s - 0.02), digits = PRECISION_DIGITS)
        push!(LB_state, min_s)
    end
    return LB_state
end

function state_space(initial_state, type = "heuristic"; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3), batched = false)
    if type == "heuristic"
        n_states = length(initial_state[1])
        LB_s = [[[] for s in 1:(n_states - 1)] for i in 1:N_REGION]
        UB_s = [[[] for s in 1:(n_states - 1)] for i in 1:N_REGION]
        for i in 1:N_REGION
            #Only need for n_states -1 states because the value of the last state is determined by the value of the 5 firsts
            for s in 1:(n_states - 1)
                LB_s[i][s] = generate_min_bound_state(initial_state, s, i, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
                UB_s[i][s] = generate_max_bound_state(initial_state, s, i, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            end
        end   
        state_iterator = Iterators.product()
        values_possible = 10 ^ PRECISION_DIGITS
        state_space_set = []
        number_of_states = 0
        #Can be done in //
        for i in 1:N_REGION
            state_i = []
            for time in 1:(T+1)
                state_i_time = []
                state_iterator = Iterators.product([(LB_s[i][s][time] * values_possible ):(UB_s[i][s][time] * values_possible) for s in 1:(n_states-1)]...)
                for state_comb in state_iterator
                    if sum(state_comb) <= values_possible + EPSILON
                        vlast = values_possible - sum(state_comb)
                        output = collect(state_comb ./ values_possible)
                        push!(output, vlast ./ values_possible)
                        push!(state_i_time, output)
                        number_of_states += 1
                    end
                end
                push!(state_i, state_i_time)
            end
            push!(state_space_set, state_i)
        end
        state_store_next = [[[Dict(zip(TREATMENT_VALS,repeat([[]], length(TREATMENT_VALS)))) for s in 1:length(state_space_set[i][t])] for t in 1:T] for i in 1:N_REGION]
        state_store_cost = [[[Dict(zip(TREATMENT_VALS,repeat([0.0], length(TREATMENT_VALS)))) for s in 1:length(state_space_set[i][t])] for t in 1:T] for i in 1:N_REGION]
        @showprogress for i in 1:N_REGION
            flush(stdout)
            count_sim = 0
            for t in T:-1:1
                for s in 1:length(state_space_set[i][t])
                    for var_comb in TREATMENT_VALS
                        # for each state and decision, find the next state
                        count_sim += 1
                        sol_states, cost = solve_ode(state_space_set[i][t][s], var_comb, i, t, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
                        cost = cost * WEIGHT[i]
                        state_dist = [norm(sol_states .- state_space_set[i][t+1][ind], Inf) for ind in 1:length(state_space_set[i][t+1])]
                        next_val, next = findmin(state_dist)
                        if batched
                            if next_val > BATCHED_NORM_BOUND
                                error("Batch error bound not respected")
                            end
                        end                    
                        # store the next state
                        state_store_next[i][t][s][var_comb] = [next]
                        # store the cost of the decision
                        state_store_cost[i][t][s][var_comb] = cost_function(state_space_set[i][t+1][next], i, t)
                    end
                end
            end
            println("Region ", i, " runs ", count_sim, " simulations")
        end
        return state_space_set, number_of_states, state_store_next, state_store_cost
    elseif type == "naive"
        n_states = length(initial_state[1])
        number_of_states = 0
        state_space_set = []
        # new_possible_states_dict_master = Dict()
        state_store_next = []
        state_store_cost = []
        for i in 1:N_REGION
            state_i = []
            push!(state_store_next, [])
            push!(state_store_cost, [])
            state_i_time = [initial_state[i]]
            new_possible_states = [initial_state[i]]
            possible_states = []
            push!(state_i, state_i_time)
            number_of_states += 1
            # new_possible_states_dict_master[i] = Dict()
            for t in 2:(T+1)
                state_i_time = []
                possible_states = new_possible_states
                new_possible_states = []
                push!(state_store_next[i], [Dict(zip(TREATMENT_VALS,repeat([[]], length(TREATMENT_VALS)))) for s in 1:length(possible_states)])
                push!(state_store_cost[i], [Dict(zip(TREATMENT_VALS,repeat([0.0], length(TREATMENT_VALS)))) for s in 1:length(possible_states)])
                num_new_states = 0
                new_possible_state_ind = 0
                new_possible_states_dict = Dict()
                for var_comb in TREATMENT_VALS
                    for (count, possible_state) in enumerate(possible_states)
                        sol_states, cost = solve_ode(possible_state, var_comb, i, t-1, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)                     
                        cur_state = [x for x in sol_states]
                        if batched
                            if num_new_states > 0
                                val, idx = findmin([max(norm(cur_state .- new_possible_states_dict[i][2], Inf), 
                                norm(cur_state .- new_possible_states_dict[i][3], Inf)) for i in 1:num_new_states])
                                if val < BATCHED_NORM_BOUND
                                    sum_state, min_state, max_state, num_points, counts = new_possible_states_dict[idx]
                                    sum_state = sum_state .+ cur_state
                                    min_state = min.(min_state, cur_state)
                                    max_state = max.(max_state, cur_state)
                                    num_points = num_points + 1
                                    push!(counts, (count, var_comb))
                                    new_possible_states_dict[idx] = (sum_state, min_state, max_state, num_points, counts)
                                else
                                    num_new_states = num_new_states + 1
                                    new_possible_states_dict[num_new_states] = (cur_state,cur_state,cur_state, 1, [(count, var_comb)])
                                    number_of_states += 1
                                end
                            else
                                num_new_states = num_new_states + 1
                                new_possible_states_dict[num_new_states] = (cur_state,cur_state,cur_state, 1, [(count, var_comb)])
                                number_of_states += 1
                            end                                                           
                        else
                            push!(new_possible_states, cur_state)
                            push!(state_i_time, cur_state)
                            new_possible_state_ind = new_possible_state_ind + 1
                            state_store_next[i][t-1][count][var_comb] = [new_possible_state_ind]
                            state_store_cost[i][t-1][count][var_comb] = cost_function(cur_state, i, t)
                            number_of_states += 1
                        end
                    end
                end
                if batched
                    for l in 1:num_new_states
                        sum_state, min_state, max_state, num_points, counts = new_possible_states_dict[l]
                        average_state = sum_state ./ num_points
                        average_state_rounded = ode_solution_rounding(average_state)
                        push!(new_possible_states, average_state_rounded)
                        push!(state_i_time, average_state_rounded)
                        new_possible_state_ind = new_possible_state_ind + 1
                        for (count, var_comb) in counts
                            state_store_next[i][t-1][count][var_comb] = [l]
                            state_store_cost[i][t-1][count][var_comb] = cost_function(average_state_rounded, i, t)
                        end
                    end
                    # new_possible_states_dict_master[i][t] = new_possible_states_dict
                end
                push!(state_i, state_i_time)
            end
            push!(state_space_set, state_i)
        end
        return state_space_set, number_of_states, state_store_next, state_store_cost
    else
        return NaN, NaN, NaN, NaN
    end
end         

#Compute best and worst solutions for modified cost functions
function compute_best_and_worst(state_space, store_next_state, store_state_cost)
    worst = []
    best = []
    for i in 1:N_REGION
        tem_cost_w = 0
        tem_cost_b = 0
        next_w = [1]
        next_b = [1]
        for t in 1:T
            # this is not very general
            # NOTE: This cost likely needs to be fixed
            decision_w = Tuple(repeat([0], N_DECISION))
            decision_b = Tuple(repeat([num_treatment_vals], N_DECISION))
            # store the next state
            next_new_w = store_next_state[i][t][next_w[1]][decision_w]
            next_new_b = store_next_state[i][t][next_b[1]][decision_b]
            # store the cost of the decision
            tem_cost_temp_w = store_state_cost[i][t][next_w[1]][decision_w]
            tem_cost_temp_b = store_state_cost[i][t][next_b[1]][decision_b]

            tem_cost_w += tem_cost_temp_w
            tem_cost_b += tem_cost_temp_b
            next_w = next_new_w
            next_b = next_new_b
        end
        push!(worst, tem_cost_w)
        push!(best, tem_cost_b)
    end
    return worst, best
end

function DP_All_sim(state_space; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3), batched = false)
    state_store_next = [[[Dict(zip(TREATMENT_VALS,repeat([[]], length(TREATMENT_VALS)))) for s in 1:length(state_space[i][t])] for t in 1:T] for i in 1:N_REGION]
    state_store_cost = [[[Dict(zip(TREATMENT_VALS,repeat([0.0], length(TREATMENT_VALS)))) for s in 1:length(state_space[i][t])] for t in 1:T] for i in 1:N_REGION]
    @showprogress for i in 1:N_REGION
        flush(stdout)
        count_sim = 0
        for t in T:-1:1
            for s in 1:length(state_space[i][t])
                for var_comb in TREATMENT_VALS
                    # for each state and decision, find the next state
                    count_sim += 1
                    sol_states, cost = solve_ode(state_space[i][t][s], var_comb, i, t, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
                    cost = cost * WEIGHT[i]
                    state_dist = [norm(sol_states .- state_space[i][t+1][ind], Inf) for ind in 1:length(state_space[i][t+1])]
                    next_val, next = findmin(state_dist)
                    if batched
                        if next_val > BATCHED_NORM_BOUND
                            error("Batch error bound not respected")
                        end
                    end                    
                    # store the next state
                    state_store_next[i][t][s][var_comb] = [next]
                    # store the cost of the decision
                    state_store_cost[i][t][s][var_comb] = cost_function(state_space[i][t+1][next], i, t)
                end
            end
        end
        println("Region ", i, " runs ", count_sim, " simulations")
    end
    return state_store_next, state_store_cost
end

#General Model with facilities
function RMP_define_model()
    RMP = Model(Gurobi.Optimizer)
    set_optimizer_attribute(RMP, "OutputFlag", 0)

    z_p = @variable(RMP, 0 <= z[1:(RECORD_END-1)])
    w = @variable(RMP, 0 <= w[1:FACILITY_LOCATIONS] <= 1)
    v = @variable(RMP, 0 <= v[i=1:N_REGION, j=1:FACILITY_LOCATIONS, t=1:T])

    #Facility Constraints
    @constraint(RMP, F_BUDGET, sum(w[j] for j in 1:FACILITY_LOCATIONS) == F)

    #=
    #Cluster Constraints
    for clust in 1:length(facilities_cluster_10)
        @constraint(RMP, sum(w[j] for j in facilities_cluster_10[clust]) >= BRANCHING_RANGE_CLUST_10[clust][1])
        @constraint(RMP, sum(w[j] for j in facilities_cluster_10[clust]) <= BRANCHING_RANGE_CLUST_10[clust][2])
    end
    =#

    #Capacity Constraint
    #=
    for t in 1:T
        for j in 1:FACILITY_LOCATIONS
            indexes = []
            for i in 1:N_REGION
                if j in FACILITIES[i]
                    push!(indexes, findfirst(x -> x == j, FACILITIES[i]))
                else
                    push!(indexes, 0)
                end
            end
        @constraint(RMP, sum(v[i,indexes[i],t] for i in 1:N_REGION if j in FACILITIES[i]) <= w[j]*CAPACITY[j])
        end
    end
    =#
    @constraint(RMP, F_CAPACITY[j=1:FACILITY_LOCATIONS, t=1:T], sum(v[i,j,t] for i in 1:N_REGION) <= CAPACITY[j]*w[j])
    @constraint(RMP, BUDGET[t=1:T], sum(sum(v[i,j,t] for j in 1:FACILITY_LOCATIONS) for i in 1:N_REGION) <= vaccine_budget)
    
    return RMP, z_p, w, v
end

#Model with threshold for distance
function RMP_define_model_travel()
    RMP = Model(Gurobi.Optimizer)
    set_optimizer_attribute(RMP, "OutputFlag", 0)
    set_optimizer_attribute(RMP, "Threads", Threads.nthreads())

    z_p = @variable(RMP, 0 <= z[1:(RECORD_END-1)])
    w = @variable(RMP, 0 <= w[1:FACILITY_LOCATIONS] <= 1)
    v = @variable(RMP, 0 <= v[i=1:N_REGION, j=1:length(FACILITIES[i]), t=1:T])

    #Facility constraint
    @constraint(RMP,F_BUDGET, sum(w[j] for j in 1:FACILITY_LOCATIONS) == F)

    #Total Budget Constraint
    @constraint(RMP, TOT_BUDGET[t=1:T], sum(sum(v[i,j,t] for j in 1:length(FACILITIES[i])) for i in 1:N_REGION) <= vaccine_budget)

    #Capacity Constraint
    for t in 1:T
        for j in 1:FACILITY_LOCATIONS
            indexes = []
            for i in 1:N_REGION
                if j in FACILITIES[i]
                    push!(indexes, findfirst(x -> x == j, FACILITIES[i]))
                else
                    push!(indexes, 0)
                end
            end
            #A facility cannot deliver more than its capacity
            @constraint(RMP, sum(v[i,indexes[i],t] for i in 1:N_REGION if j in FACILITIES[i]) <= w[j]*CAPACITY[j])
        end
    end

    #You cannot vaccinate more people than there is in one state
    for j in 1:FACILITY_LOCATIONS
        indexes = []
        for i in 1:N_REGION
            if j in FACILITIES[i]
                push!(indexes, findfirst(x -> x == j, FACILITIES[i]))
                @constraint(RMP, sum(v[i,indexes[i],t] for t in 1:T) <= POPULATION[i][indexes[i]])
                for t in 1:T
                    @constraint(RMP, v[i,indexes[i],t] <= w[j]*min(POPULATION[i][indexes[i]], CAPACITY[j]))
                end
            else
                push!(indexes, 0)
            end
        end
    end

    #@constraint(RMP, F_CAPACITY[j=1:FACILITY_LOCATIONS, t=1:T], sum(v[i,j,t] for i in 1:N_REGION if j in FACILITIES[i]) <= CAPACITY[j]*w[j])
    @constraint(RMP, BUDGET[t=1:T], sum(sum(v[i,j,t] for j in 1:length(FACILITIES[i])) for i in 1:N_REGION) <= vaccine_budget)

    #Remove infeasible facilities
    for j in 1:FACILITY_LOCATIONS
        if FACILITY_INFEAS_IND[j]
            @constraint(RMP, w[j] <= 0)
            continue
        end
        if FACILITY_MUST_IND[j]
            @constraint(RMP, w[j] == 1)
        end
    end

    return RMP, z_p, w, v
end

#General Model
function RMP_add_columns(model, z, C_CONSISTENT, C_REGION, w, v)
    println("RMP begins: Currently, the total number of plans is ", RECORD_END - 1, " from ", RECORD_NOW - 1)
    #We update the constraints of our model
    if ALREADY_OPTIMIZED
        for p in RECORD_NOW:(RECORD_END-1)
            push!(z, @variable(model, base_name = "z[$(p)]", lower_bound = 0, upper_bound = Inf))
            for i in 1:N_REGION
                if i == PLAN_REGION_IND[p]
                    for t in 1:T
                        for decision in 1:N_DECISION
                            # update budget cost
                            set_normalized_coefficient(C_CONSISTENT[i, decision, t], z[p], DECISION_PT[p][t][decision])
                        end
                    end
                end
            end
            for i in 1:N_REGION
                if PLAN_REGION_IND[p] == i
                    # add plan into consideration
                    set_normalized_coefficient(C_REGION[i], z[p], 1)
                end
            end
        end 
    else
        #Budget constraint
        @constraint(model, C_CONSISTENT[i=1:N_REGION, d=1:N_DECISION, t=1:T], sum(z[p]*DECISION_PT[p][t][d] for p in 1:(RECORD_END-1) if PLAN_REGION_IND[p] == i) == sum(v[i,j,t] for j in 1:length(FACILITIES[i]))/vaccine_unit)
        
        #One plan per region
        @constraint(model, C_REGION[i=1:N_REGION], sum(z[p] for p in 1:(RECORD_END-1) if PLAN_REGION_IND[p] == i) == 1)
    end
        
    #Remove infeasible plans
    for p in RECORD_NOW:(RECORD_END-1)
        if PLAN_INFEAS_IND[p]
            @constraint(model, z[p] <= 0)
            continue
        end
    end

    #We minimize the wieghted cost
    #println("here 1")
    @objective(model, Min, sum(WEIGHT[PLAN_REGION_IND[p]]*z[p]*COST_P[p] for p in 1:(RECORD_END-1)))
    #+ TRAVEL_WEIGHT*sum(sum(sum(v[i,j,t]*COST_DIST[i][j] for t in 1:T) for j in 1:length(FACILITIES[i])) for i in 1:N_REGION))
    #println("here 2")
    global RECORD_NOW = deepcopy(RECORD_END)
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        Total_cost = objective_value(model)
        println("The optimal objective is ", Total_cost)
        
        #Retrieve dual variables
        pi_budget = [[[dual(C_CONSISTENT[i,d,t]) for t in 1:T] for d in 1:N_DECISION] for i in 1:N_REGION]
        phi = [dual(C_REGION[i]) for i in 1:N_REGION]

        #Check reduced costs
        #=
        for p in 1:RECORD_END-1
            real_rc = reduced_cost(z[p])
            region_p = PLAN_REGION_IND[p]
            hand_rc = COST_P[p] - phi[region_p] - sum(sum(DECISION_PT[p][t][d]*pi_budget[region_p][d][t] for d in 1:N_DECISION) for t in 1:T)
            if (real_rc != hand_rc) & (abs(hand_rc-real_rc) > 1e-3)
                println("Problem for plan ", p, " in region  ", PLAN_REGION_IND[p], ": ",  DECISION_PT[p])
                println("Value for this plan is: ", value.(z[p]))
                println("This plan is infeasible:", PLAN_INFEAS_IND[p])
                println("Real reduced cost is ", real_rc, " and hand is ", hand_rc)
            end
        end
        =#
    else
        pi_budget = [[[0.0 for t in 1:T] for d in 1:N_DECISION] for i in 1:N_REGION]
        phi = [0.0 for i in 1:N_REGION]
        Total_cost = Inf
        println("No Feasible Solution for Gurobi!!!")
    end
    
    #Important for CG
    global NEGATIVE_COLUMN = false
    #Tells you the first time you optimized the full model
    global ALREADY_OPTIMIZED = true
    
    return model, z, pi_budget, phi, C_CONSISTENT, C_REGION, Total_cost
end

"""
    Backward_DP(region, initial_state, pi_budget, phi, state_space, state_store_next, state_store_cost)
    Solve DP problem as part of pricing subproblem backwards from time for one region
"""

function Backward_DP(region, initial_state, pi_budget, phi, state_space, state_store_next, state_store_cost; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3))
    #We initilize the costs
    state_cost = [[Inf for s in 1:length(state_space[region][t])] for t in 1:(T+1)]
    state_backward = [[0.0 for s in 1:length(state_space[region][t])] for t in 1:T]
    state_decision = [[Tuple(repeat([0.0], N_DECISION)) for s in 1:length(state_space[region][t])] for t in 1:T]
    
    #final state costs
    for s in 1:length(state_space[region][T+1])
        state_cost[T+1][s] = -phi[region]
    end
    #We go backward to find the plan with the most negative reduced cost
    for t in T:-1:1
        for s in 1:length(state_space[region][t])
            min_cost = Inf
            min_state = -1
            min_var_comb = Tuple(repeat([NaN], N_DECISION))
            for var_comb in TREATMENT_VALS
                # We check that the branching constraints are satisfied
                branching_unsatisfied = 0
                for (idx, val) in enumerate(var_comb)
                    branching_unsatisfied = branching_unsatisfied | (val < BRANCHING_RANGE[region][t][idx][1]) | (val > BRANCHING_RANGE[region][t][idx][2])
                end
                if branching_unsatisfied == 1
                    continue
                end

                # we retrieve the next state for this decision_combination and state
                tem_cost = state_store_cost[region][t][s][var_comb]
                tem_next = state_store_next[region][t][s][var_comb]
                if length(tem_next) == 0
                    continue
                end
                # We compute the reduced cost of this decision: the cost is the sum of decision cost plus next state cost minus reduced costs
                tem_cost = tem_cost + state_cost[t+1][tem_next[1]] - sum([pi_budget[region][d][t]*var_comb[d] for d in 1:N_DECISION])
                # keep track of minimum costs
                if tem_cost < min_cost
                    min_cost = tem_cost
                    min_state = tem_next[1]
                    min_var_comb = var_comb
                end
            end
            #Keep in memory the best decision for each state
            state_cost[t][s] = min_cost
            state_backward[t][s] = min_state
            state_decision[t][s] = min_var_comb
        end
    end
    
    #We now compute the best decision starting from the initial state
    state_opt = 1
    temp_cost = 0
    actual_cost = 0
    decision = [Tuple(repeat([0], N_DECISION)) for t in 1:T]
    
    #Check that there is a column with a negative reduced cost
    region_reduced_cost = state_cost[1][state_opt]
    #println("The reduced cost of region $region is $region_reduced_cost")
    if region_reduced_cost < -EPSILON
        #Go forward to store the best decision
        for t in 1:T
            decision[t] = Int.(state_decision[t][state_opt])
            temp_cost += state_store_cost[region][t][state_opt][decision[t]]
            state_opt = state_backward[t][state_opt]
            state_opt = Int(state_opt)
        end  
        # #Now we compute the actual cost of the selected plan
        # temp_state = initial_state[region]
        # for t in 1:T
        #     temp_state, cost_t = solve_ode(temp_state, decision[t], region, t, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
        #     temp_cost += cost_t
        # end
    end
    return decision, temp_cost
end     

#DP with modified costs
function Backward_DP_modified_costs(region, initial_state, pi_budget, phi, state_space, state_store_next, state_store_cost, worst, best; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3))
    #We initilize the costs
    state_cost = [[Inf for s in 1:length(state_space[region][t])] for t in 1:(T+1)]
    state_backward = [[0.0 for s in 1:length(state_space[region][t])] for t in 1:T]
    state_decision = [[Tuple(repeat([0.0], N_DECISION)) for s in 1:length(state_space[region][t])] for t in 1:T]
    
    #final state costs
    for s in 1:length(state_space[region][T+1])
        state_cost[T+1][s] = -phi[region]
    end
    #We go backward to find the plan with the most negative reduced cost
    for t in T:-1:1
        for s in 1:length(state_space[region][t])
            min_cost = Inf
            min_state = -1
            min_var_comb = Tuple(repeat([NaN], N_DECISION))
            for var_comb in TREATMENT_VALS
                # We check that the branching constraints are satisfied
                branching_unsatisfied = 0
                for (idx, val) in enumerate(var_comb)
                    branching_unsatisfied = branching_unsatisfied | (val < BRANCHING_RANGE[region][t][idx][1]) | (val > BRANCHING_RANGE[region][t][idx][2])
                end
                if branching_unsatisfied == 1
                    continue
                end
                # we retrieve the next state for this decision_combination and state
                tem_cost = state_store_cost[region][t][s][var_comb]
                tem_next = state_store_next[region][t][s][var_comb]

                if t == T
                    tem_cost = (best[region] - tem_cost)/(best[region]-worst[region])
                end

                if length(tem_next) == 0
                    continue
                end
                # We compute the reduced cost of this decision: the cost is the sum of decision cost plus next state cost minus reduced costs
                tem_cost = tem_cost + state_cost[t+1][tem_next[1]] - sum([pi_budget[region][d][t]*var_comb[d] for d in 1:N_DECISION])
                # keep track of minimum costs
                if tem_cost < min_cost
                    min_cost = tem_cost
                    min_state = tem_next[1]
                    min_var_comb = var_comb
                end
            end
            #Keep in memory the best decision for each state
            state_cost[t][s] = min_cost
            state_backward[t][s] = min_state
            state_decision[t][s] = min_var_comb
        end
    end
    
    #We now compute the best decision starting from the initial state
    state_opt = 1
    temp_cost = 0
    actual_cost = 0
    decision = [Tuple(repeat([0], N_DECISION)) for t in 1:T]
    
    #Check that there is a column with a negative reduced cost
    region_reduced_cost = state_cost[1][state_opt]
    # println("the reduced cost of region $region is $region_reduced_cost")
    if region_reduced_cost < -EPSILON
        #Go forward to store the best decision
        for t in 1:T
            decision[t] = Int.(state_decision[t][state_opt])
            temp_cost += state_store_cost[region][t][state_opt][decision[t]]
            state_opt = state_backward[t][state_opt]
            state_opt = Int(state_opt)
        end  
        # #Now we compute the actual cost of the selected plan
        # temp_state = initial_state[region]
        # for t in 1:T
        #     temp_state, cost_t = solve_ode(temp_state, decision[t], region, t, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
        #     temp_cost += cost_t
        # end
    end
    return decision, temp_cost
end  
"""
    BackwardAll(initial_state, pi_budget, phi, state_space, store_next_state, store_state_cost)
    Solve DP problem as part of pricing subproblem backwards from time for all regions
"""
function BackwardAll(initial_state, pi_budget, phi, state_space, store_next_state, store_state_cost; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3))
    # backward solve DP for all regions
    new_plan = []
    cost = []
    max_time = 0
    for i in 1:N_REGION #Can do it in Parallel
        (plan_i, cost_i), time_dp = @timed Backward_DP(i, initial_state, pi_budget, phi, state_space, store_next_state, store_state_cost, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
        push!(new_plan, plan_i)
        push!(cost, cost_i)
        #if time_dp > max_time
        max_time += time_dp
    end
    return new_plan, cost
end

function BackwardAll_modified_cost(initial_state, pi_budget, phi, state_space, store_next_state, store_state_cost, worst, best; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3))
    # backward solve DP for all regions
    new_plan = []
    cost = []
    max_time = 0
    for i in 1:N_REGION #Can do it in Parallel
        (plan_i, cost_i), time_dp = @timed Backward_DP_modified_costs(i, initial_state, pi_budget, phi, state_space, store_next_state, store_state_cost, worst, best, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
        push!(new_plan, plan_i)
        push!(cost, cost_i)
        #if time_dp > max_time
        max_time += time_dp
    end
    return new_plan, cost
end

function ini_columns(initial_state, state_space, store_next_state, store_state_cost; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3))
    # create dummy columns for default plans
    for i in 1:N_REGION
        PLAN_REGION_IND[RECORD_END] = i
        state_region_temp = initial_state[i]
        tem_cost = 0
        next = [1]
        for t in 1:T
            # this is not very general
            # NOTE: This cost likely needs to be fixed
            DECISION_PT[RECORD_END][t] = Tuple(repeat([0], N_DECISION))
            # store the next state
            next_new = store_next_state[i][t][next[1]][DECISION_PT[RECORD_END][t]]
            # store the cost of the decision
            tem_cost_temp = store_state_cost[i][t][next[1]][DECISION_PT[RECORD_END][t]]
            tem_cost += tem_cost_temp
            next = next_new
        end

        COST_P[RECORD_END] = 1.01*tem_cost
        PLAN_INFEAS_IND[RECORD_END] = false
        
        global RECORD_END += 1
    end

    cost_dummy = sum(COST_P[j] for j in 1:N_REGION)

    for i in 1:N_REGION
        COST_P[i] = cost_dummy
    end
    
    #Create null-plans
    for i in 1:N_REGION
        PLAN_REGION_IND[RECORD_END] = i
        state_region_temp = initial_state[i]
        tem_cost = 0
        next = [1]
        for t in 1:T
            # this is not very general
            # NOTE: This cost likely needs to be fixed
            DECISION_PT[RECORD_END][t] = Tuple(repeat([0], N_DECISION))
            # store the next state
            next_new = store_next_state[i][t][next[1]][DECISION_PT[RECORD_END][t]]
            # store the cost of the decision
            tem_cost_temp = store_state_cost[i][t][next[1]][DECISION_PT[RECORD_END][t]]
            tem_cost += tem_cost_temp
            next = next_new
        end

        COST_P[RECORD_END] = tem_cost
        PLAN_INFEAS_IND[RECORD_END] = false
        
        global RECORD_END += 1
    end

    ini_tot_cost = 0
    for i in 1:N_REGION
        PLAN_REGION_IND[RECORD_END] = i
        state_region_temp = initial_state[i]
        tem_cost = 0
        next = [1]
        for t in 1:T
            # this is not very general
            # NOTE: This cost likely needs to be fixed
            DECISION_PT[RECORD_END][t] = Tuple(repeat([1], N_DECISION))
            # store the next state
            next_new = store_next_state[i][t][next[1]][DECISION_PT[RECORD_END][t]]
            # store the cost of the decision
            tem_cost_temp = store_state_cost[i][t][next[1]][DECISION_PT[RECORD_END][t]]
            tem_cost += tem_cost_temp
            next = next_new
        end
        COST_P[RECORD_END] = tem_cost
        PLAN_INFEAS_IND[RECORD_END] = false
        
        ini_tot_cost += COST_P[RECORD_END]*WEIGHT[i]
        global RECORD_END += 1
    end
    global UB = deepcopy(ini_tot_cost)
end

function extract_column(new_plan, costs)
    for i in 1:N_REGION
        #Check that this region has a plan with negative reduced cost
        region_cost = costs[i]
        # println("the cost of region $i is $region_cost")
        PLAN_INFEAS_IND_TEMP = PLAN_INFEAS_IND[1:RECORD_END]
        # println("infeasibility indicator: $PLAN_INFEAS_IND_TEMP")
        if region_cost == 0
            continue
        end
        global NEGATIVE_COLUMN = true  #If you never pass the previous "if" loop, this value is false and it ends the CG
        PLAN_REGION_IND[RECORD_END] = i
        for t in 1:T
            DECISION_PT[RECORD_END][t] = new_plan[i][t]
        end
        #println("New plan for region ", i, " is: ", new_plan[i])
        #=
        for p in 1:RECORD_END-1
            if (new_plan[i]==DECISION_PT[p]) & (PLAN_REGION_IND[p] == i)
                #println("This plan is already in the plan set with cost: ", COST_P[p])
                #println("The new plan cost is: ", region_cost)
            end
        end
        =#
        COST_P[RECORD_END] = region_cost
        PLAN_INFEAS_IND[RECORD_END] = false
        # println("additional plan for region")        
        global RECORD_END += 1
    end
end

function column_generation(initial_state, state_space, store_next_state, store_state_cost, C_CONSISTENT, C_REGION; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3))
    model, z, w, v = RMP_define_model()
    time_dp_total = 0
    time_mp_total = 0
    obj = 0
    while true
        (model, z, pi_budget, phi, C_CONSISTENT, C_REGION, obj), time_mp = @timed RMP_add_columns(model, z, C_CONSISTENT, C_REGION, w, v)
        time_mp_total += time_mp
        (new_plan, costs), time_dp = @timed BackwardAll(initial_state, pi_budget, phi, state_space, store_next_state, store_state_cost, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
        extract_column(new_plan, costs)
        time_dp_total += time_dp
        println("-----------------------------------------------------")
        if !NEGATIVE_COLUMN
            break
        end
    end
    println("Column Generations Ends")
    println("-----------------------------------------------------")
    return model, z, C_CONSISTENT, C_REGION, time_dp_total, time_mp_total, obj
end

function reset_global_params()
    global RECORD_NOW = 1
    global RECORD_END = 1
    global NEGATIVE_COLUMN = false
    global BRANCHING_RANGE = deepcopy(GLOBAL_BRANCHING_RANGE)
    global BRANCHING_RANGE_FACILITY = deepcopy(GLOBAL_BRANCHING_RANGE_FACILITY)
    global BRANCHING_RANGE_CLUST_10 = deepcopy(GLOBAL_BRANCHING_RANGE_CLUST_10)
    global LB_NODE = [0.0 for n in 1:NCOLB]
    global PRE_NODE = [1 for n in 1:NCOLB]
    global SIGN_NODE = ["left" for n in 1:NCOLB]
    global THRE_NODE = [[1, 1, 1.0, 0] for n in 1:NCOLB]
    global CUR_NODE = 0
    global TOT_NODE = 0
    global UB = Inf
    global LB = - Inf
    global ALREADY_OPTIMIZED = false
    global C_BUDGET = []
    global C_REGION = []
    global C_CONSISTENT = []
    global F_BUDGET = []
    global F_CAPACITY = []
    global BUDGET = []
    global SHIPPED = []
    global COST_P = deepcopy(GLOBAL_COST_P)
    global DECISION_PT =  deepcopy(GLOBAL_DECISION_PT)
    global PLAN_REGION_IND =  deepcopy(GLOBAL_PLAN_REGION_IND)
    global PLAN_INFEAS_IND =  deepcopy(GLOBAL_PLAN_INFEAS_IND)
    global FACILITY_INFEAS_IND =  deepcopy(GLOBAL_FACILITY_INFEAS_IND)
    global FACILITY_MUST_IND =  deepcopy(GLOBAL_FACILITY_MUST_IND)
    global GOOD_BRANCHINGS = deepcopy(GLOBAL_GOOD_BRANCHINGS)
end

#IPs
function solve_model_ip()
    IP = Model(Gurobi.Optimizer)
    set_optimizer_attribute(IP, "OutputFlag", 0)
    set_optimizer_attribute(IP, "Threads", Threads.nthreads())

    z_ip = @variable(IP, z_ip[1:(RECORD_END-1)], Bin)
    w_ip = @variable(IP, w_ip[1:FACILITY_LOCATIONS], Bin)
    v_ip = @variable(IP, 0 <= v_ip[1:N_REGION, 1:FACILITY_LOCATIONS, 1:T])

    @constraint(IP, F_BUDGET, sum(w_ip[j] for j in 1:FACILITY_LOCATIONS) == F)
    @constraint(IP, F_CAPACITY[j=1:FACILITY_LOCATIONS, t=1:T], sum(v_ip[i,j,t] for i in 1:N_REGION) <= CAPACITY[j]*w_ip[j])
    @constraint(IP, BUDGET[t=1:T], sum(sum(v_ip[i,j,t] for i in 1:N_REGION) for j in 1:FACILITY_LOCATIONS) <= vaccine_budget)
    
    @constraint(IP, c_consistent_final[i=1:N_REGION, d=1:N_DECISION, t=1:T], sum(z_ip[p]*DECISION_PT[p][t][d] for p in 1:(RECORD_END-1) if PLAN_REGION_IND[p] == i) == sum(v_ip[i,j,t] for j in 1:FACILITY_LOCATIONS)/vaccine_unit)
    @constraint(IP, c_region_final[i=1:N_REGION], sum(z_ip[p] for p in 1:(RECORD_END-1) if PLAN_REGION_IND[p] == i) == 1)

    #=
    for p in 1:RECORD_END-1
        if PLAN_INFEAS_IND[p]
            println("Plan ", p, " is infeasible")
            @constraint(IP, z_ip[p] <= 0)
        end
    end

    for j in 1:FACILITY_LOCATIONS
        if FACILITY_INFEAS_IND[j]
            println("Facility ", j, " is infeasible")
            @constraint(model, w_ip[j] <= 0)
            continue
        end
    end
    =#
    #=
    for i in 1:N_REGION
        exists = false
        for p in 1:RECORD_END-1
            if (PLAN_REGION_IND[p] == i) & (DECISION_PT[p] == PLANS_NO_FACILITY[i])
                @constraint(IP, z_ip[p] >= 1)
                exists = true
            end
        end
        if !exists
            println("The plan does not exist for region ", i)
        end
    end 
    =#
    @objective(IP, Min, sum(WEIGHT[PLAN_REGION_IND[p]]*z_ip[p]*COST_P[p] for p in 1:(RECORD_END-1))
    + TRAVEL_WEIGHT*sum(sum(sum(v_ip[i,j,t]*COST_DIST[i][j] for t in 1:T) for j in 1:FACILITY_LOCATIONS) for i in 1:N_REGION))
    optimize!(IP)
    obj_ip = objective_value(IP)

    return IP, z_ip, obj_ip, w_ip, v_ip
end

#Solve IP with threshold
function solve_model_ip_travel()
    IP = Model(Gurobi.Optimizer)
    set_optimizer_attribute(IP, "OutputFlag", 0)
    set_optimizer_attribute(IP, "Threads", Threads.nthreads())

    z_ip = @variable(IP, z_ip[1:(RECORD_END-1)], Bin)
    w_ip = @variable(IP, w_ip[1:FACILITY_LOCATIONS], Bin)
    v_ip = @variable(IP, 0 <= v_ip[i=1:N_REGION, j=1:length(FACILITIES[i]), 1:T])

    @constraint(IP, F_BUDGET, sum(w_ip[j] for j in 1:FACILITY_LOCATIONS) == F)

    for t in 1:T
        for j in 1:FACILITY_LOCATIONS
            indexes = []
            for i in 1:N_REGION
                if j in FACILITIES[i]
                    push!(indexes, findfirst(x -> x == j, FACILITIES[i]))
                    @constraint(IP, v_ip[i,indexes[i],t] <= POPULATION[i][indexes[i]]*w_ip[j])

                else
                    push!(indexes, 0)
                end
            end
            @constraint(IP, sum(v_ip[i,indexes[i],t] for i in 1:N_REGION if j in FACILITIES[i]) <= w_ip[j]*CAPACITY[j])
        end
    end

    #You cannot vaccinate more people than there is in one state
    for j in 1:FACILITY_LOCATIONS
        indexes = []
        for i in 1:N_REGION
            if j in FACILITIES[i]
                push!(indexes, findfirst(x -> x == j, FACILITIES[i]))
                @constraint(IP, sum(v_ip[i,indexes[i],t] for t in 1:T) <= POPULATION[i][indexes[i]])
                for t in 1:T
                    @constraint(IP, v_ip[i,indexes[i],t] <= w_ip[j]*min(POPULATION[i][indexes[i]], CAPACITY[j]))
                end
            else
                push!(indexes, 0)
            end
        end
    end
    #@constraint(IP, F_CAPACITY[j=1:FACILITY_LOCATIONS, t=1:T], sum(v_ip[i,j,t] for i in 1:N_REGION) <= CAPACITY[j]*w_ip[j])
    @constraint(IP, BUDGET[t=1:T], sum(sum(v_ip[i,j,t] for j in 1:length(FACILITIES[i])) for i in 1:N_REGION) <= vaccine_budget)
    
    @constraint(IP, c_consistent_final[i=1:N_REGION, d=1:N_DECISION, t=1:T], sum(z_ip[p]*DECISION_PT[p][t][d] for p in 1:(RECORD_END-1) if PLAN_REGION_IND[p] == i) == sum(v_ip[i,j,t] for j in 1:length(FACILITIES[i]))/vaccine_unit)
    @constraint(IP, c_region_final[i=1:N_REGION], sum(z_ip[p] for p in 1:(RECORD_END-1) if PLAN_REGION_IND[p] == i) == 1)

    @objective(IP, Min, sum(WEIGHT[PLAN_REGION_IND[p]]*z_ip[p]*COST_P[p] for p in 1:(RECORD_END-1)))
    #+ TRAVEL_WEIGHT*sum(sum(sum(v_ip[i,j,t]*COST_DIST[i][j] for t in 1:T) for j in 1:FACILITY_LOCATIONS) for i in 1:N_REGION))
    optimize!(IP)
   
    if (termination_status(IP) == MOI.INFEASIBLE) | (termination_status(IP) == MOI.INFEASIBLE_OR_UNBOUNDED)
        obj_ip = Inf
    else
        obj_ip = objective_value(IP)
    end
    return IP, z_ip, obj_ip, w_ip, v_ip
end

#Solve IP with threshold and rounding facilities
function solve_model_ip_travel_round(facilities_yes, facilities_no)

    IP = Model(Gurobi.Optimizer)
    set_optimizer_attribute(IP, "OutputFlag", 0)
    set_optimizer_attribute(IP, "Threads", Threads.nthreads())

    z_ip = @variable(IP, z_ip[1:(RECORD_END-1)], Bin)
    w_ip = @variable(IP, w_ip[1:FACILITY_LOCATIONS], Bin)
    v_ip = @variable(IP, 0 <= v_ip[i=1:N_REGION, j=1:length(FACILITIES[i]), 1:T])

    @constraint(IP, F_BUDGET, sum(w_ip[j] for j in 1:FACILITY_LOCATIONS) == F)
    #Total Budget Constraint
    @constraint(IP, TOT_BUDGET[t=1:T], sum(sum(v_ip[i,j,t] for j in 1:length(FACILITIES[i])) for i in 1:N_REGION) <= vaccine_budget)

    #Forcing the facilities by rounding the LP
    for j in 1:FACILITY_LOCATIONS
        if j in facilities_yes
            @constraint(IP, w_ip[j] == 1)
        elseif j in facilities_no
            @constraint(IP, w_ip[j] == 0)
        end
    end

    for t in 1:T
        for j in 1:FACILITY_LOCATIONS
            indexes = []
            for i in 1:N_REGION
                if j in FACILITIES[i]
                    push!(indexes, findfirst(x -> x == j, FACILITIES[i]))
                else
                    push!(indexes, 0)
                end
            end
            @constraint(IP, sum(v_ip[i,indexes[i],t] for i in 1:N_REGION if j in FACILITIES[i]) <= w_ip[j]*CAPACITY[j])
        end
    end

    #You cannot vaccinate more people than there is in one state
    for j in 1:FACILITY_LOCATIONS
        indexes = []
        for i in 1:N_REGION
            if j in FACILITIES[i]
                push!(indexes, findfirst(x -> x == j, FACILITIES[i]))
                @constraint(IP, sum(v_ip[i,indexes[i],t] for t in 1:T) <= POPULATION[i][indexes[i]]*w_ip[j])
                for t in 1:T
                    @constraint(IP, v_ip[i,indexes[i],t] <= w_ip[j]*min(POPULATION[i][indexes[i]], CAPACITY[j]))
                end
            else
                push!(indexes, 0)
            end
        end
    end

    #@constraint(IP, F_CAPACITY[j=1:FACILITY_LOCATIONS, t=1:T], sum(v_ip[i,j,t] for i in 1:N_REGION) <= CAPACITY[j]*w_ip[j])
    @constraint(IP, BUDGET[t=1:T], sum(sum(v_ip[i,j,t] for j in 1:length(FACILITIES[i])) for i in 1:N_REGION) <= vaccine_budget)
    
    @constraint(IP, c_consistent_final[i=1:N_REGION, d=1:N_DECISION, t=1:T], sum(z_ip[p]*DECISION_PT[p][t][d] for p in 1:(RECORD_END-1) if PLAN_REGION_IND[p] == i) == sum(v_ip[i,j,t] for j in 1:length(FACILITIES[i]))/vaccine_unit)
    @constraint(IP, c_region_final[i=1:N_REGION], sum(z_ip[p] for p in 1:(RECORD_END-1) if PLAN_REGION_IND[p] == i) == 1)

    @objective(IP, Min, sum(WEIGHT[PLAN_REGION_IND[p]]*z_ip[p]*COST_P[p] for p in 1:(RECORD_END-1)))
    #+ TRAVEL_WEIGHT*sum(sum(sum(v_ip[i,j,t]*COST_DIST[i][j] for t in 1:T) for j in 1:FACILITY_LOCATIONS) for i in 1:N_REGION))

    optimize!(IP)

    if (termination_status(IP) == MOI.INFEASIBLE) | (termination_status(IP) == MOI.INFEASIBLE_OR_UNBOUNDED)
        obj_ip = Inf
    else
        obj_ip = objective_value(IP)
    end

    return IP, z_ip, obj_ip, w_ip, v_ip
end

#CG Functions
function column_generation_travel(initial_state, state_space, store_next_state, store_state_cost, C_CONSISTENT, C_REGION; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3))
    #threshold = THRESHOLD_DISTANCE
    model, z, w, v = RMP_define_model_travel()
    time_dp_total = 0
    time_mp_total = 0
    obj = 0
    iterations = 0
    while true
        iterations += 1
        (model, z, pi_budget, phi, C_CONSISTENT, C_REGION, obj), time_mp = @timed RMP_add_columns(model, z, C_CONSISTENT, C_REGION, w, v)
        if termination_status(model) == MOI.INFEASIBLE
            break
        end
        time_mp_total += time_mp
        (new_plan, costs), time_dp  = @timed BackwardAll(initial_state, pi_budget, phi, state_space, store_next_state, store_state_cost, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
        time_dp_total += time_dp
        extract_column(new_plan, costs)
        println("-----------------------------------------------------")
        if !NEGATIVE_COLUMN
            break
        end
    end
    println("Column Generations Ends")
    println("-----------------------------------------------------")

    return model, z, C_CONSISTENT, C_REGION, obj, time_dp_total, time_mp_total, w, v
end

function column_generation_with_ip_travel(initial_state, state_space, store_next_state, store_state_cost, C_CONSISTENT, C_REGION; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3))
    model, z, w, v = RMP_define_model_travel()
    time_dp_total = 0
    time_mp_total = 0
    obj = 0
    iterations = 0
    while true
        iterations += 1
        (model, z, pi_budget, phi, C_CONSISTENT, C_REGION, obj), time_mp = @timed RMP_add_columns(model, z, C_CONSISTENT, C_REGION, w, v)
        if termination_status(model) == MOI.INFEASIBLE
            break
        end
        time_mp_total += time_mp
        (new_plan, costs), time_dp  = @timed BackwardAll(initial_state, pi_budget, phi, state_space, store_next_state, store_state_cost, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
        time_dp_total += time_dp
        extract_column(new_plan, costs)
        println("-----------------------------------------------------")
        if !NEGATIVE_COLUMN
            break
        end
    end
    println("Column Generations Ends")
    println("-----------------------------------------------------")
    println("Solving the IP Travel Threshold")
    
    #Simply solve the full IP model at the end of the CG iteration
    if termination_status(model) == MOI.INFEASIBLE
        IP, z_final, obj_ip, w_ip, v_ip = 0, 0, 0, 0, 0
        allocations = []
    else
        IP, z_final, obj_ip, w_ip, v_ip = solve_model_ip_travel()
        println("Objective Value is: ", obj_ip)
        if (termination_status(IP) == MOI.INFEASIBLE) | (termination_status(IP) == MOI.INFEASIBLE_OR_UNBOUNDED)
            allocations = []
        else
            allocations = []
            for i in 1:N_REGION
                for p in 1:(RECORD_END-1)
                    if (value.(z_final[p]) > EPSILON) & (PLAN_REGION_IND[p] == i)
                        #println("Region ", i, " plan: ", DECISION_PT[p])
                        push!(allocations, DECISION_PT[p])
                    end
                end
            end
        end
    end     
    return model, z, C_CONSISTENT, C_REGION, obj, obj_ip, IP, z_final, allocations, time_dp_total, time_mp_total, w, v, w_ip, v_ip
end

#CGIP with threshold and rounding facilities in IP
function column_generation_with_ip_travel_round(initial_state, state_space, store_next_state, store_state_cost, C_CONSISTENT, C_REGION, lim_1, lim_2; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3))
    #threshold = THRESHOLD_DISTANCE
    model, z, w, v = RMP_define_model_travel()
    time_dp_total = 0
    time_mp_total = 0
    obj = 0
    iterations = 0
    while true
        iterations += 1
        (model, z, pi_budget, phi, C_CONSISTENT, C_REGION, obj), time_mp = @timed RMP_add_columns(model, z, C_CONSISTENT, C_REGION, w, v)
        if termination_status(model) == MOI.INFEASIBLE
            break
        end
        time_mp_total += time_mp
        (new_plan, costs), time_dp  = @timed BackwardAll(initial_state, pi_budget, phi, state_space, store_next_state, store_state_cost, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
        time_dp_total += time_dp
        extract_column(new_plan, costs)
        println("-----------------------------------------------------")
        if !NEGATIVE_COLUMN
            break
        end
    end
    println("Column Generations Ends")
    println("-----------------------------------------------------")

    println("Solving the IP Travel Threshold")
    
    #Simply solve the full IP model at the end of the CG iteration
    if termination_status(model) == MOI.INFEASIBLE
        IP, z_final, obj_ip, w_ip, v_ip = 0, 0, 0, 0, 0
        time_ip = 0
        allocations = []
    else
        #We select some facilities
        must = []
        must_not = []
        #=
        for j in 1:FACILITY_LOCATIONS
            if value.(w[j]) < lim_1
                push!(must_not,j)
            elseif value.(w[j]) > lim_2
                push!(must,j)
            end
        end
        =#
        #println("Must Facilities: ", must, " number: ", length(must))
        #println("Must Not Facilities: ", must_not, " number: ", length(must_not))
        if lim_2 > 0
            values = [value.(w[j]) for j in 1:FACILITY_LOCATIONS]
            indexes = [j for j in 1:FACILITY_LOCATIONS]
            perm = sortperm(values)
            indexes = indexes[perm]
            must = indexes[end-F+1:end]
        end

        #Then we solve the IP by rounding facilities
        (IP, z_final, obj_ip, w_ip, v_ip), time_ip = @timed solve_model_ip_travel_round(must, must_not)
        println("Objective Value is: ", obj_ip)
        if (termination_status(IP) == MOI.INFEASIBLE) | (termination_status(IP) == MOI.INFEASIBLE_OR_UNBOUNDED)
            allocations = []
        else
            allocations = []
            for i in 1:N_REGION
                for p in 1:(RECORD_END-1)
                    if (value.(z_final[p]) > EPSILON) & (PLAN_REGION_IND[p] == i)
                        #println("Region ", i, " plan: ", DECISION_PT[p])
                        push!(allocations, DECISION_PT[p])
                    end
                end
            end
        end
    end     
    return model, z, C_CONSISTENT, C_REGION, obj, obj_ip, IP, z_final, allocations, time_dp_total, time_mp_total, w, v, w_ip, v_ip, time_ip
end

############################### BP section ###############################

#Branch on vaccines
function check_fractional_sol(model, z_pi, worst)
    fra_region = -1
    fra_time = -1
    fra_lbval = -1
    fra_decision = "No Decision"
    if termination_status(model) == MOI.OPTIMAL
        global LB = sum(worst) - objective_value(model)
        assign_it = [[repeat([0.0], N_DECISION) for t in 1:T] for i in 1:N_REGION]
        
        # Look at the value the CG assigned to each plan in the end
        # NOTE: Check with Martin
        for p in 1:(RECORD_END)
            if PLAN_INFEAS_IND[p]
                continue
            end
            current = value.(z_pi[p])
            if current > EPSILON
                for t in 1:T
                    for d in 1:N_DECISION
                        assign_it[PLAN_REGION_IND[p]][t][d] = assign_it[PLAN_REGION_IND[p]][t][d] + DECISION_PT[p][t][d]*current
                    end
                end
            end
        end
        
        #Select the most fractional value, the region, the time period, and the x or y decision associated with it
        close_gap = 0.5
        middle_point = 0.5
        #For x
        for d in 1:N_DECISION
            for i in 1:N_REGION
                for t in 1:T
                    if ((assign_it[i][t][d]%1) > EPSILON) & ((assign_it[i][t][d]%1) < (1 - EPSILON))
                        if abs((assign_it[i][t][d]%1) - middle_point) < close_gap
                            close_gap = abs((assign_it[i][t][d]%1) - middle_point)
                            fra_region = i
                            fra_time = t
                            fra_lbval = floor(assign_it[i][t][d])
                            fra_decision = d
                        end
                    end
                end
            end
        end
    else
        println("No feasible solution for GUROBI")
    end
    println("Branching: Region ", fra_region, " period ", fra_time, " floor ", fra_lbval, " on ", fra_decision)
    return (Int(round(fra_region)), Int(round(fra_time)), fra_lbval, fra_decision, assign_it)
end

#Tri-branching
function check_fractional_sol_z(model, z_pi, worst)
    fra_region = -1
    fra_time = -1
    fra_lbval = -1
    fra_decision = "No Decision"
    if termination_status(model) == MOI.OPTIMAL
        global LB = sum(worst) - objective_value(model)
        assign_it = [[repeat([0.0], N_DECISION) for t in 1:T] for i in 1:N_REGION]
                    
        # Look at the value the CG assigned to each plan in the end
        # NOTE: Check with Martin
        for p in 1:(RECORD_END)
            if PLAN_INFEAS_IND[p]
                continue
            end
            current = value.(z_pi[p])
            if current > EPSILON
                for t in 1:T
                    for d in 1:N_DECISION
                        assign_it[PLAN_REGION_IND[p]][t][d] = assign_it[PLAN_REGION_IND[p]][t][d] + DECISION_PT[p][t][d]*current
                    end
                end
            end
        end

        #Select the time, region and decision that leads to the highest difference in term of plans
        #println("After checking infeasibility")
        max_diff = 0
        for t in 1:T
            for i in 1:N_REGION
                for d in 1:N_DECISION
                    cur_val = 0
                    for p in 1:(RECORD_END)
                        if PLAN_INFEAS_IND[p]
                            continue 
                        end
                        if PLAN_REGION_IND[p] == i
                            cur_val += value.(z_pi[p])*abs(assign_it[PLAN_REGION_IND[p]][t][d] - DECISION_PT[p][t][d])
                        end
                    end
                    if cur_val > max_diff
                        max_diff = cur_val
                        fra_region = i
                        fra_time = t
                        fra_lbval = round(assign_it[i][t][d])
                        fra_decision = d
                    end
                end
            end
        end
        println("Max Diff: ", max_diff)
    else
        println("No feasible solution for GUROBI")
    end
    #println("Branching: Region ", fra_region, " period ", fra_time, " floor ", fra_lbval, " on ", fra_decision)
    return (Int(round(fra_region)), Int(round(fra_time)), fra_lbval, fra_decision, assign_it)
end

#Branch on single facilities
function check_fractional_sol_w(model, w, worst)
    fra_facility = -1
    if termination_status(model) == MOI.OPTIMAL
        close_gap = 0.5
        middle_point = 0.5
        global LB = sum(worst) - objective_value(model)
        if termination_status(model) == MOI.OPTIMAL
            for j in 1:FACILITY_LOCATIONS
                if (value.(w[j]) > EPSILON) & (value.(w[j]) < 1 - EPSILON)
                    if abs(value.(w[j]) - middle_point) < close_gap
                        close_gap = abs(value.(w[j]) - middle_point)
                        fra_facility = j
                    end
                end
            end
        end
    end
    println("Branching on Facility: ", fra_facility)

    return (Int(round(fra_facility)))
end

#Left Branching
function left_branching()
    fra_region = Int(round(THRE_NODE[CUR_NODE][1]))
    fra_time = Int(round(THRE_NODE[CUR_NODE][2]))
    fra_lbval = THRE_NODE[CUR_NODE][3]
    fra_decision = Int(round(THRE_NODE[CUR_NODE][4]))
    
    #We add one left branching constraint
    if fra_decision > 0
        BRANCHING_RANGE[fra_region][fra_time][fra_decision][2] = min(fra_lbval, BRANCHING_RANGE[fra_region][fra_time][fra_decision][2])
        if BRANCHING_RANGE[fra_region][fra_time][fra_decision][2] < BRANCHING_RANGE[fra_region][fra_time][fra_decision][1]
            println("Warnings: Infeasible branching constraints on variable $fra_decision. Left")
        end
    end
    
    #We prune the columns
    for p in 1:(RECORD_END-1)
        #Do not do it for the dummy plans
        if p < N_REGION + 1
            continue
        end
        
        #Only for the region that is concerned by the new constraint
        if PLAN_REGION_IND[p] != fra_region
            continue
        end
        infeasibility_ind = 0
        for d in 1:N_DECISION
            infeasibility_ind = infeasibility_ind | (DECISION_PT[p][fra_time][d] > BRANCHING_RANGE[fra_region][fra_time][d][2])
        end
        if infeasibility_ind == 1
            PLAN_INFEAS_IND[p] = true
        end
    end
end

#LB branchings
function best_LB_branching()
    global BRANCHING_RANGE = deepcopy(GLOBAL_BRANCHING_RANGE)
    global BRANCHING_RANGE_FACILITY = deepcopy(GLOBAL_BRANCHING_RANGE_FACILITY)
    global BRANCHING_RANGE_CLUST_10 = deepcopy(GLOBAL_BRANCHING_RANGE_CLUST_10)
    back_node = CUR_NODE
    
    #We need to add a list of new constraints on the range of vehicles available
    while back_node != 0
        fra_region = Int(round(THRE_NODE[back_node][1]))
        fra_time = Int(round(THRE_NODE[back_node][2]))
        fra_lbval = THRE_NODE[back_node][3]
        fra_sign = SIGN_NODE[back_node]
        fra_decision = Int(round(THRE_NODE[back_node][4]))

        #Means we are branching on a normal decision variable
        if (fra_decision > 0) & (fra_decision < N_DECISION + 1)
            if fra_sign == "left"
                BRANCHING_RANGE[fra_region][fra_time][fra_decision][2] = min(fra_lbval, BRANCHING_RANGE[fra_region][fra_time][fra_decision][2]) #We add the left branching constraint
            elseif fra_sign == "right"
                BRANCHING_RANGE[fra_region][fra_time][fra_decision][1] = max(fra_lbval, BRANCHING_RANGE[fra_region][fra_time][fra_decision][1]) #We add the right branching constraint
            elseif fra_sign == "middle"
                BRANCHING_RANGE[fra_region][fra_time][fra_decision][1] = fra_lbval
                BRANCHING_RANGE[fra_region][fra_time][fra_decision][2] = fra_lbval
            end
            if BRANCHING_RANGE[fra_region][fra_time][fra_decision][2] < BRANCHING_RANGE[fra_region][fra_time][fra_decision][1] # We check that our two branching constraints are feasible
                println("Warnings: Infeasible Branching constraints on x. BestLB")
                println("Region ", fra_region, " at time ", fra_time, " on value ", fra_lbval, " with sign ", fra_sign)
                println(BRANCHING_RANGE[fra_region][fra_time][fra_decision][2])
                println(BRANCHING_RANGE[fra_region][fra_time][fra_decision][1])
                global GOOD_BRANCHINGS = false
            end
        end 

        #Means we are branching on a facility location variable
        if fra_decision == 0
            if fra_sign == "left"
                BRANCHING_RANGE_FACILITY[fra_region] = 0
            elseif fra_sign == "right"
                BRANCHING_RANGE_FACILITY[fra_region] = 1
            end
        end

        #Means we are branching on 10-clusters
        if fra_decision == 10
            if fra_sign == "left"
                BRANCHING_RANGE_CLUST_10[fra_region][2] = min(fra_lbval, BRANCHING_RANGE_CLUST_10[fra_region][2])
            elseif fra_sign == "right"
                BRANCHING_RANGE_CLUST_10[fra_region][1] = max(fra_lbval, BRANCHING_RANGE_CLUST_10[fra_region][1])
            end
            if BRANCHING_RANGE_CLUST_10[fra_region][2] < BRANCHING_RANGE_CLUST_10[fra_region][1]
                println("Warnings: Infeasible Branching constraints on clusters.")
                println("Cluster ", fra_region, " on value ", fra_lbval)
                println(BRANCHING_RANGE_CLUST_10[fra_region][2])
                println(BRANCHING_RANGE_CLUST_10[fra_region][1])
                GOOD_BRANCHINGS = false
            end
        end
        back_node = PRE_NODE[back_node]
    end
    
    #Now we prune the columns for the plans
    for p in 1:(RECORD_END-1)
        PLAN_INFEAS_IND[p] = false
        #Done to avoid doing it for the dummy columns at the beginning
        if p < N_REGION + 1
            continue
        end
        #Check wether or not the old plans are still feasible with the new constraints
        i_R = PLAN_REGION_IND[p]
        for t in 1:T
            infeasibility_ind = 0
            for d in 1:N_DECISION
                #infeasibility_ind = infeasibility_ind | (DECISION_PT[p][t][d] < BRANCHING_RANGE[i_R][t][d][1]) | (DECISION_PT[p][t][d] > BRANCHING_RANGE[i_R][t][d][2])
                if (DECISION_PT[p][t][d] < BRANCHING_RANGE[i_R][t][d][1]) | (DECISION_PT[p][t][d] > BRANCHING_RANGE[i_R][t][d][2])
                    infeasibility_ind = 1
                    break
                end
            end
            if infeasibility_ind == 1
                PLAN_INFEAS_IND[p] = true
                break
            end
        end
    end

    #Now we add the infeasibilies for each facilities
    for j in 1:FACILITY_LOCATIONS
        FACILITY_INFEAS_IND[j] = false
        FACILITY_MUST_IND[j] = false
        if BRANCHING_RANGE_FACILITY[j] == 0
            FACILITY_INFEAS_IND[j] = true
        end
        if BRANCHING_RANGE_FACILITY[j] == 1
            FACILITY_MUST_IND[j] = true
        end
        if (FACILITY_MUST_IND[j]) & (FACILITY_INFEAS_IND[j])
            println("Infeasible constraints for facilities in facility ", j)
            global GOOD_BRANCHINGS = false
        end
    end 
end

#BP_z on facilities
#BP_z on facilities
function branch_and_price_group(initial_state, state_space, next_states, state_costs, ip, lim_1, lim_2; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3), upper_bound = Inf, optimality_gap = 0)
    BB_nodes = [0]
    LB_leaves = []
    best_assign_it = [[repeat([0], N_DECISION) for t in 1:T] for i in 1:N_REGION]
    global C_CONSISTENT = []
    global C_REGION = []
    reset_global_params()
    worst, best = compute_best_and_worst(state_space, next_states, state_costs)
    ini_columns(initial_state, state_space, next_states, state_costs; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3))
    model, z, w, v = RMP_define_model_travel()
    best_model, best_z = model, z
    z_final = z
    nb_cg = 0
    nb_plans = [0.0]
    global UB = upper_bound
    t_mp_total = 0
    t_dp_total = 0
    t_cg_total = 0
    t_ip_total = 0
    left_branch = 0
    best_lb_branch = 0
    obj_ip = Inf
    feasible = true
    prune_by_bound = 0
    prune_by_infeas = 0
    branch_w = 0
    branch_x = 0
    branch_z = 0
    best_w = [0.0 for j in 1:FACILITY_LOCATIONS]
    start_time = time()
    end_time = time()
    elapsed_time = end_time - start_time
    while (true) & (elapsed_time < 3600*3)
        #if TOT_NODE >= 9990
        #    break
        #end
        popfirst!(BB_nodes)
        #Start by solving CG a first time
        global C_BUDGET = []
        global C_CONSISTENT = []
        global C_REGION = []

        (model, z, C_CONSISTENT, C_REGION, obj, obj_ip, IP, z_final, allocations, t_dp, t_mp,  w, v, w_ip, v_ip, t_ip), t_cg = @timed column_generation_with_ip_travel_round(initial_state, state_space, next_states, state_costs, C_CONSISTENT, C_REGION, lim_1, 2, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
        
        t_mp_total += t_mp
        t_dp_total += t_dp
        t_cg_total += t_cg
        t_ip_total += t_ip 
        nb_cg += 1
        push!(nb_plans, RECORD_END-1)
        #Check if the problem is feasible or not (either we selected a dummy plan or simply infeasible)
        if (termination_status(model) == MOI.INFEASIBLE) | (termination_status(model) == MOI.INFEASIBLE_OR_UNBOUNDED)
            feasible = false
            prune_by_infeas += 1
            #println("Infeasible")
        else
            for p in 1:N_REGION
                if value.(z[p]) > 0.0001
                    feasible = false
                    println("This node is infeasible")
                    prune_by_infeas +=1
                    break
                end
            end
        end

        #Select the most fractional solution
        fra_region_w = check_fractional_sol_w(model, w, worst)
        fra_region_x = fra_region_w
        println("BPz T = ", T, " D = ", num_treatment_vals)
        if (LB-UB)/LB <= optimality_gap
            push!(LB_leaves, LB)
            prune_by_bound += 1
        end

        #Branching on Facilities
        if (((LB-UB)/LB) > optimality_gap) & (fra_region_w > 0) & (feasible) #Solution is still fractional   
            if fra_region_w > 0
                branch_w += 1
                #We add left node first
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "left"
                THRE_NODE[TOT_NODE][1] = fra_region_w
                THRE_NODE[TOT_NODE][2] = 0
                THRE_NODE[TOT_NODE][3] = 0.0
                THRE_NODE[TOT_NODE][4] = 0

                if length(BB_nodes) == 0
                    push!(BB_nodes, TOT_NODE)
                else
                    for s in 1:length(BB_nodes)
                        if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                            if s == length(BB_nodes)
                                push!(BB_nodes, TOT_NODE)
                            end
                            continue
                        else
                            insert!(BB_nodes, s, TOT_NODE)
                            break
                        end
                    end
                end

                #We then add the right node
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "right"
                THRE_NODE[TOT_NODE][1] = fra_region_w
                THRE_NODE[TOT_NODE][2] = 0
                THRE_NODE[TOT_NODE][3] = 1.0
                THRE_NODE[TOT_NODE][4] = 0

                for s in 1:length(BB_nodes)
                    if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                        if s == length(BB_nodes)
                            push!(BB_nodes, TOT_NODE)
                        end
                        continue
                    else
                        insert!(BB_nodes, s, TOT_NODE)
                        break
                    end
                end
            end
        end
        
        #Branching on vaccines
        if (((LB-UB)/LB) > optimality_gap) & (fra_region_w < 0) & (feasible) #Facilities are integer
            fra_region_x, fra_time_x, fra_lbval_x, fra_decision_x, assign_it_x = check_fractional_sol(model, z, worst)
            if fra_region_x > 0 #Solution is still fractional for vaccines
                branch_x += 1
                #We add left node first
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "left"
                THRE_NODE[TOT_NODE][1] = fra_region_x
                THRE_NODE[TOT_NODE][2] = fra_time_x
                THRE_NODE[TOT_NODE][3] = fra_lbval_x
                THRE_NODE[TOT_NODE][4] = fra_decision_x

                if length(BB_nodes) == 0
                    push!(BB_nodes, TOT_NODE)
                else
                    for s in 1:length(BB_nodes)
                        if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                            if s == length(BB_nodes)
                                push!(BB_nodes, TOT_NODE)
                            end
                            continue
                        else
                            insert!(BB_nodes, s, TOT_NODE)
                            break
                        end
                    end
                end
    
                #We then add the right node
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "right"
                THRE_NODE[TOT_NODE][1] = fra_region_x
                THRE_NODE[TOT_NODE][2] = fra_time_x
                THRE_NODE[TOT_NODE][3] = fra_lbval_x + 1
                THRE_NODE[TOT_NODE][4] = fra_decision_x
                
                for s in 1:length(BB_nodes)
                    if s == length(BB_nodes)
                        push!(BB_nodes, TOT_NODE)
                        break
                    elseif LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                        continue
                    else
                        insert!(BB_nodes, s, TOT_NODE)
                        break
                    end
                end
            end
        end

        #If you saved more lives you update the UB
        if (sum(worst) - deepcopy(obj_ip) > UB) & (feasible)
            global UB = sum(worst) - deepcopy(obj_ip)
            println("Update UB: ", UB)
            best_assign_it = allocations
            for j in 1:FACILITY_LOCATIONS
                if value.(w_ip[j]) > 0.01
                    best_w[j] = 1.0
                else 
                    best_w[j] = 0.0
                end
            end
        end
        println("Nodes in the queue ", length(BB_nodes), " with UB ", UB)
        
        if length(BB_nodes) > 0 #While we still have some nodes on the tree
            if upper_bound < Inf
                global CUR_NODE = BB_nodes[1]
                best_LB_branching()
                println("Proceed Best LB branching! Current node is ", CUR_NODE, " with LB: ", LB_NODE[CUR_NODE])
                println("The current gap is: ", (LB_NODE[CUR_NODE]-UB)/LB)
                best_lb_branch += 1                
            else
                if PRE_NODE[BB_nodes[1]] == CUR_NODE #Means we are going for the left branching
                    global CUR_NODE = BB_nodes[1]
                    left_branching()
                    println("Proceed left Branching!")
                    left_branch += 1
                else #Means that we are going for the lower bound branching
                    global CUR_NODE = BB_nodes[1]
                    best_LB_branching()
                    println("Proceed Best LB branching!")
                    best_lb_branch += 1
                end
            end
        else
            break
        end

        if !GOOD_BRANCHINGS
            println("Branching issue")
            break
        end
        global RECORD_NOW = 1
        global ALREADY_OPTIMIZED = false
        feasible = true
        end_time = time()
        elapsed_time = end_time - start_time
    end

    if length(BB_nodes) > 0
        for leaves in BB_nodes
            push!(LB_leaves, LB_NODE[leaves])
        end
    end

    IP_final, z_final, obj_ip_final, w_final, v_final = solve_model_ip_travel_round([],[])
    UB_IP_final = sum(worst) - deepcopy(obj_ip_final)

    allocations = best_assign_it

    return best_assign_it, z, allocations, z_final, t_dp_total, t_cg_total, t_mp_total, best_lb_branch, TOT_NODE, LB_leaves, UB, prune_by_infeas, prune_by_bound, branch_w, branch_x, branch_z, best_w, t_ip_total, UB_IP_final
end

function branch_and_price_z_group(initial_state, state_space, next_states, state_costs, ip, lim_1, lim_2; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3), upper_bound = Inf, optimality_gap = 0)
    BB_nodes = [0]
    LB_leaves = []
    best_assign_it = [[repeat([0], N_DECISION) for t in 1:T] for i in 1:N_REGION]
    global C_CONSISTENT = []
    global C_REGION = []
    reset_global_params()
    worst, best = compute_best_and_worst(state_space, next_states, state_costs)
    ini_columns(initial_state, state_space, next_states, state_costs; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3))
    model, z, w, v = RMP_define_model_travel()
    best_model, best_z = model, z
    z_final = z
    nb_cg = 0
    nb_plans = [0.0]
    global UB = upper_bound
    t_mp_total = 0
    t_dp_total = 0
    t_cg_total = 0
    t_ip_total = 0
    left_branch = 0
    best_lb_branch = 0
    obj_ip = Inf
    feasible = true
    prune_by_bound = 0
    prune_by_infeas = 0
    branch_w = 0
    branch_x = 0
    branch_z = 0
    best_w = [0.0 for j in 1:FACILITY_LOCATIONS]
    start_time = time()
    end_time = time()
    elapsed_time = end_time - start_time
    while (true) & (elapsed_time < 3600*3)

        popfirst!(BB_nodes)
        #Start by solving CG a first time
        global C_BUDGET = []
        global C_CONSISTENT = []
        global C_REGION = []

        (model, z, C_CONSISTENT, C_REGION, obj, obj_ip, IP, z_final, allocations, t_dp, t_mp,  w, v, w_ip, v_ip, t_ip), t_cg = @timed column_generation_with_ip_travel_round(initial_state, state_space, next_states, state_costs, C_CONSISTENT, C_REGION, lim_1, 2, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
        
        t_mp_total += t_mp
        t_dp_total += t_dp
        t_cg_total += t_cg
        t_ip_total += t_ip 
        nb_cg += 1
        push!(nb_plans, RECORD_END-1)
        #Check if the problem is feasible or not (either we selected a dummy plan or simply infeasible)
        if (termination_status(model) == MOI.INFEASIBLE) | (termination_status(model) == MOI.INFEASIBLE_OR_UNBOUNDED)
            feasible = false
            prune_by_infeas += 1
            #println("Infeasible")
        else
            for p in 1:N_REGION
                if value.(z[p]) > 0.0001
                    feasible = false
                    println("This node is infeasible")
                    prune_by_infeas +=1
                    break
                end
            end
        end

        #Select the most fractional solution
        fra_region_w = check_fractional_sol_w(model, w, worst)
        fra_region_x = fra_region_w
        println("BPz T = ", T, " D = ", num_treatment_vals)
        if (LB-UB)/LB <= optimality_gap
            push!(LB_leaves, LB)
            prune_by_bound += 1
        end

        #Branching on Facilities
        if (((LB-UB)/LB) > optimality_gap) & (fra_region_w > 0) & (feasible) #Solution is still fractional   
            if fra_region_w > 0
                branch_w += 1
                #We add left node first
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "left"
                THRE_NODE[TOT_NODE][1] = fra_region_w
                THRE_NODE[TOT_NODE][2] = 0
                THRE_NODE[TOT_NODE][3] = 0.0
                THRE_NODE[TOT_NODE][4] = 0

                if length(BB_nodes) == 0
                    push!(BB_nodes, TOT_NODE)
                else
                    for s in 1:length(BB_nodes)
                        if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                            if s == length(BB_nodes)
                                push!(BB_nodes, TOT_NODE)
                            end
                            continue
                        else
                            insert!(BB_nodes, s, TOT_NODE)
                            break
                        end
                    end
                end

                #We then add the right node
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "right"
                THRE_NODE[TOT_NODE][1] = fra_region_w
                THRE_NODE[TOT_NODE][2] = 0
                THRE_NODE[TOT_NODE][3] = 1.0
                THRE_NODE[TOT_NODE][4] = 0

                for s in 1:length(BB_nodes)
                    if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                        if s == length(BB_nodes)
                            push!(BB_nodes, TOT_NODE)
                        end
                        continue
                    else
                        insert!(BB_nodes, s, TOT_NODE)
                        break
                    end
                end
            end
        end
        
        #Branching on vaccines
        if (((LB-UB)/LB) > optimality_gap) & (fra_region_w < 0) & (feasible) #Facilities are integer
            fra_region_x, fra_time_x, fra_lbval_x, fra_decision_x, assign_it_x = check_fractional_sol(model, z, worst)
            if fra_region_x > 0 #Solution is still fractional for vaccines
                branch_x += 1
                #We add left node first
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "left"
                THRE_NODE[TOT_NODE][1] = fra_region_x
                THRE_NODE[TOT_NODE][2] = fra_time_x
                THRE_NODE[TOT_NODE][3] = fra_lbval_x
                THRE_NODE[TOT_NODE][4] = fra_decision_x

                if length(BB_nodes) == 0
                    push!(BB_nodes, TOT_NODE)
                else
                    for s in 1:length(BB_nodes)
                        if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                            if s == length(BB_nodes)
                                push!(BB_nodes, TOT_NODE)
                            end
                            continue
                        else
                            insert!(BB_nodes, s, TOT_NODE)
                            break
                        end
                    end
                end
    
                #We then add the right node
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "right"
                THRE_NODE[TOT_NODE][1] = fra_region_x
                THRE_NODE[TOT_NODE][2] = fra_time_x
                THRE_NODE[TOT_NODE][3] = fra_lbval_x + 1
                THRE_NODE[TOT_NODE][4] = fra_decision_x
                
                for s in 1:length(BB_nodes)
                    if s == length(BB_nodes)
                        push!(BB_nodes, TOT_NODE)
                        break
                    elseif LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                        continue
                    else
                        insert!(BB_nodes, s, TOT_NODE)
                        break
                    end
                end
            end
        end
        
        #Branching on plans
        if (((LB-UB)/LB) > optimality_gap)  & (fra_region_x < 0) & (feasible) #Integer solution is found in x
            println("Branching for z")
            #Check if the solution is fractional in terms of z
            fra_region_z, fra_time_z, fra_lbval_z, fra_decision_z, assign_it = check_fractional_sol_z(model, z, worst)
            best_model, best_z = IP, z_final
            if (fra_region_z < 0) #Means that the solution is also integer in terms of z
                println("Integer for z")
                push!(LB_leaves, LB)
                #best_assign_it = allocations
                #=
                for i in 1:N_REGION
                    for t in 1:T
                        for d in 1:N_DECISION
                            best_assign_it[i][t][d] = round(assign_it[i][t][d])
                        end
                    end
                end
                =#

            else
                branch_z += 1
                println("Branching region: ", fra_region_z, " in time ", fra_time_z, " with value ", fra_lbval_z, " on variable ", fra_decision_z)
                #We add left node first
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "left"
                THRE_NODE[TOT_NODE][1] = fra_region_z
                THRE_NODE[TOT_NODE][2] = fra_time_z
                THRE_NODE[TOT_NODE][3] = fra_lbval_z - 1
                THRE_NODE[TOT_NODE][4] = fra_decision_z

                if length(BB_nodes) == 0
                    push!(BB_nodes, TOT_NODE)
                else
                    for s in 1:length(BB_nodes)
                        if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                            if s == length(BB_nodes)
                                push!(BB_nodes, TOT_NODE)
                            end
                            continue
                        else
                            insert!(BB_nodes, s, TOT_NODE)
                            break
                        end
                    end
                end

                #We then add the middle node
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "middle"
                THRE_NODE[TOT_NODE][1] = fra_region_z
                THRE_NODE[TOT_NODE][2] = fra_time_z
                THRE_NODE[TOT_NODE][3] = fra_lbval_z
                THRE_NODE[TOT_NODE][4] = fra_decision_z

                for s in 1:length(BB_nodes)
                    if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                        if s == length(BB_nodes)
                            push!(BB_nodes, TOT_NODE)
                        end
                        continue
                    else
                        insert!(BB_nodes, s, TOT_NODE)
                        break
                    end
                end

                #We finally add the right node
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "right"
                THRE_NODE[TOT_NODE][1] = fra_region_z
                THRE_NODE[TOT_NODE][2] = fra_time_z
                THRE_NODE[TOT_NODE][3] = fra_lbval_z + 1
                THRE_NODE[TOT_NODE][4] = fra_decision_z
                
                for s in 1:length(BB_nodes)
                    if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                        if s == length(BB_nodes)
                            push!(BB_nodes, TOT_NODE)
                        end
                        continue
                    else
                        insert!(BB_nodes, s, TOT_NODE)
                        break
                    end
                end
            end
        end
        

        #If you saved more lives you update the UB
        if (sum(worst) - deepcopy(obj_ip) > UB) & (feasible)
            global UB = sum(worst) - deepcopy(obj_ip)
            println("Update UB: ", UB)
            best_assign_it = allocations
            for j in 1:FACILITY_LOCATIONS
                if value.(w_ip[j]) > 0.01
                    best_w[j] = 1.0
                else 
                    best_w[j] = 0.0
                end
            end
        end
        println("Nodes in the queue ", length(BB_nodes), " with UB ", UB)
        
        if length(BB_nodes) > 0 #While we still have some nodes on the tree
            if upper_bound < Inf
                global CUR_NODE = BB_nodes[1]
                best_LB_branching()
                println("Proceed Best LB branching! Current node is ", CUR_NODE, " with LB: ", LB_NODE[CUR_NODE])
                println("The current gap is: ", (LB_NODE[CUR_NODE]-UB)/LB)
                best_lb_branch += 1                
            else
                if PRE_NODE[BB_nodes[1]] == CUR_NODE #Means we are going for the left branching
                    global CUR_NODE = BB_nodes[1]
                    left_branching()
                    println("Proceed left Branching!")
                    left_branch += 1
                else #Means that we are going for the lower bound branching
                    global CUR_NODE = BB_nodes[1]
                    best_LB_branching()
                    println("Proceed Best LB branching!")
                    best_lb_branch += 1
                end
            end
        else
            break
        end

        if !GOOD_BRANCHINGS
            println("Branching issue")
            break
        end
        global RECORD_NOW = 1
        global ALREADY_OPTIMIZED = false
        feasible = true
        end_time = time()
        elapsed_time = end_time - start_time
    end

    if length(BB_nodes) > 0
        for leaves in BB_nodes
            push!(LB_leaves, LB_NODE[leaves])
        end
    end

    IP_final, z_final, obj_ip_final, w_final, v_final = solve_model_ip_travel_round([],[])
    UB_IP_final = sum(worst) - deepcopy(obj_ip_final)

    allocations = best_assign_it

    return best_assign_it, z, allocations, z_final, t_dp_total, t_cg_total, t_mp_total, best_lb_branch, TOT_NODE, LB_leaves, UB, prune_by_infeas, prune_by_bound, branch_w, branch_x, branch_z, best_w, t_ip_total, UB_IP_final
end

function branch_and_price_z_group_small(initial_state, state_space, next_states, state_costs, ip, lim1, lim2; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3), upper_bound = Inf, optimality_gap = 0)
    BB_nodes = [0]
    LB_leaves = []
    best_assign_it = [[repeat([0], N_DECISION) for t in 1:T] for i in 1:N_REGION]
    global C_CONSISTENT = []
    global C_REGION = []
    reset_global_params()
    worst, best = compute_best_and_worst(state_space, next_states, state_costs)
    ini_columns(initial_state, state_space, next_states, state_costs; ode_accuracy = :auto, ode_accuracy_level = (1e-3, 1e-3))
    model, z = RMP_define_model()
    best_model, best_z = model, z
    z_final = z
    nb_cg = 0
    nb_plans = [0.0]
    global UB = upper_bound
    t_mp_total = 0
    t_dp_total = 0
    t_cg_total = 0
    left_branch = 0
    best_lb_branch = 0
    obj_ip = Inf
    feasible = true
    prune_by_bound = 0
    prune_by_infeas = 0
    while true
        if TOT_NODE >= 3
            break
        end
        popfirst!(BB_nodes)
        #Start by solving CG a first time
        global C_BUDGET = []
        global C_CONSISTENT = []
        global C_REGION = []

        (model, z, C_CONSISTENT, C_REGION, obj, obj_ip, IP, z_final, allocations, t_dp, t_mp,  w, v, w_ip, v_ip, t_ip), t_cg = @timed column_generation_with_ip_travel_round(initial_state, state_space, next_states, state_costs, C_CONSISTENT, C_REGION, lim1, lim2, ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
        
        t_mp_total += t_mp
        t_dp_total += t_dp
        t_cg_total += t_cg
        nb_cg += 1
        push!(nb_plans, RECORD_END-1)

        #Check if the problem is feasible or not (either we selected a dummy plan or simply infeasible)
        if (termination_status(model) == MOI.INFEASIBLE) | (termination_status(model) == MOI.INFEASIBLE_OR_UNBOUNDED)
            feasible = false
            prune_by_infeas += 1
        else
            for p in 1:N_REGION
                if value.(z[p]) > 0.0001
                    feasible = false
                    println("This node is infeasible")
                    prune_by_infeas +=1
                    break
                end
            end
        end

        #Select the most fractional solution
        fra_region_w = check_fractional_sol_w(model, w, worst)
        fra_region_x = fra_region_w
        println("BPz T = ", T, " D = ", num_treatment_vals)
        if (LB-UB)/LB <= optimality_gap
            push!(LB_leaves, LB)
            prune_by_bound += 1
        end
        #Branching on 10-cluster facilities

        #Branching on Facilities
        if (((LB-UB)/LB) > optimality_gap) & (fra_region_w > 0) & (feasible) #Solution is still fractional   
            if fra_region_w > 0
                #We add left node first
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "left"
                THRE_NODE[TOT_NODE][1] = fra_region_w
                THRE_NODE[TOT_NODE][2] = 0
                THRE_NODE[TOT_NODE][3] = 0.0
                THRE_NODE[TOT_NODE][4] = 0

                if length(BB_nodes) == 0
                    push!(BB_nodes, TOT_NODE)
                else
                    for s in 1:length(BB_nodes)
                        if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                            if s == length(BB_nodes)
                                push!(BB_nodes, TOT_NODE)
                            end
                            continue
                        else
                            insert!(BB_nodes, s, TOT_NODE)
                            break
                        end
                    end
                end

                #We then add the right node
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "right"
                THRE_NODE[TOT_NODE][1] = fra_region_w
                THRE_NODE[TOT_NODE][2] = 0
                THRE_NODE[TOT_NODE][3] = 1.0
                THRE_NODE[TOT_NODE][4] = 0

                for s in 1:length(BB_nodes)
                    if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                        if s == length(BB_nodes)
                            push!(BB_nodes, TOT_NODE)
                        end
                        continue
                    else
                        insert!(BB_nodes, s, TOT_NODE)
                        break
                    end
                end
            end
        end
        
        #Branching on vaccines
        if (((LB-UB)/LB) > optimality_gap) & (fra_region_w < 0) & (feasible) #Facilities are integer
            fra_region_x, fra_time_x, fra_lbval_x, fra_decision_x, assign_it_x = check_fractional_sol(model, z, worst)
            if fra_region_x > 0 #Solution is still fractional for vaccines
                #We add left node first
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "left"
                THRE_NODE[TOT_NODE][1] = fra_region_x
                THRE_NODE[TOT_NODE][2] = fra_time_x
                THRE_NODE[TOT_NODE][3] = fra_lbval_x
                THRE_NODE[TOT_NODE][4] = fra_decision_x

                if length(BB_nodes) == 0
                    push!(BB_nodes, TOT_NODE)
                else
                    for s in 1:length(BB_nodes)
                        if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                            if s == length(BB_nodes)
                                push!(BB_nodes, TOT_NODE)
                            end
                            continue
                        else
                            insert!(BB_nodes, s, TOT_NODE)
                            break
                        end
                    end
                end
    
                #We then add the right node
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "right"
                THRE_NODE[TOT_NODE][1] = fra_region_x
                THRE_NODE[TOT_NODE][2] = fra_time_x
                THRE_NODE[TOT_NODE][3] = fra_lbval_x + 1
                THRE_NODE[TOT_NODE][4] = fra_decision_x
                
                for s in 1:length(BB_nodes)
                    if s == length(BB_nodes)
                        push!(BB_nodes, TOT_NODE)
                        break
                    elseif LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                        continue
                    else
                        insert!(BB_nodes, s, TOT_NODE)
                        break
                    end
                end
            end
        end
        
        #Branching on plans
        if (((LB-UB)/LB) > optimality_gap)  & (fra_region_x < 0) & (feasible) #Integer solution is found in x
            println("Branching for z")
            #Check if the solution is fractional in terms of z
            fra_region_z, fra_time_z, fra_lbval_z, fra_decision_z, assign_it = check_fractional_sol_z(model, z, worst)
            best_model, best_z = IP, z_final
            if (fra_region_z < 0) & (UB <= LB) #Means that the solution is also integer in terms of z
                println("Integer for z")
                push!(LB_leaves, LB)
                for i in 1:N_REGION
                    for t in 1:T
                        for d in 1:N_DECISION
                            best_assign_it[i][t][d] = round(assign_it[i][t][d])
                        end
                    end
                end

            else
                println("Branching region: ", fra_region_z, " in time ", fra_time_z, " with value ", fra_lbval_z, " on variable ", fra_decision_z)
                #We add left node first
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "left"
                THRE_NODE[TOT_NODE][1] = fra_region_z
                THRE_NODE[TOT_NODE][2] = fra_time_z
                THRE_NODE[TOT_NODE][3] = fra_lbval_z - 1
                THRE_NODE[TOT_NODE][4] = fra_decision_z

                if length(BB_nodes) == 0
                    push!(BB_nodes, TOT_NODE)
                else
                    for s in 1:length(BB_nodes)
                        if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                            if s == length(BB_nodes)
                                push!(BB_nodes, TOT_NODE)
                            end
                            continue
                        else
                            insert!(BB_nodes, s, TOT_NODE)
                            break
                        end
                    end
                end

                #We then add the middle node
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "middle"
                THRE_NODE[TOT_NODE][1] = fra_region_z
                THRE_NODE[TOT_NODE][2] = fra_time_z
                THRE_NODE[TOT_NODE][3] = fra_lbval_z
                THRE_NODE[TOT_NODE][4] = fra_decision_z

                for s in 1:length(BB_nodes)
                    if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                        if s == length(BB_nodes)
                            push!(BB_nodes, TOT_NODE)
                        end
                        continue
                    else
                        insert!(BB_nodes, s, TOT_NODE)
                        break
                    end
                end

                #We finally add the right node
                global TOT_NODE += 1
                LB_NODE[TOT_NODE] = LB
                PRE_NODE[TOT_NODE] = CUR_NODE
                SIGN_NODE[TOT_NODE] = "right"
                THRE_NODE[TOT_NODE][1] = fra_region_z
                THRE_NODE[TOT_NODE][2] = fra_time_z
                THRE_NODE[TOT_NODE][3] = fra_lbval_z + 1
                THRE_NODE[TOT_NODE][4] = fra_decision_z
                
                for s in 1:length(BB_nodes)
                    if LB_NODE[TOT_NODE] <= LB_NODE[BB_nodes[s]]
                        if s == length(BB_nodes)
                            push!(BB_nodes, TOT_NODE)
                        end
                        continue
                    else
                        insert!(BB_nodes, s, TOT_NODE)
                        break
                    end
                end
            end
        end
        

        #If you saved more lives you update the UB
        if sum(worst) - deepcopy(obj_ip) > UB
            global UB = sum(worst) - deepcopy(obj_ip)
            println("Update UB: ", UB)
            best_assign_it = allocations
        end
        println("Nodes in the queue ", length(BB_nodes), " with UB ", UB)
        
        if length(BB_nodes) > 0 #While we still have some nodes on the tree
            if upper_bound < Inf
                global CUR_NODE = BB_nodes[1]
                best_LB_branching()
                println("Proceed Best LB branching! Current node is ", CUR_NODE, " with LB: ", LB_NODE[CUR_NODE])
                println("The current gap is: ", (LB_NODE[CUR_NODE]-UB)/LB)
                best_lb_branch += 1                
            else
                if PRE_NODE[BB_nodes[1]] == CUR_NODE #Means we are going for the left branching
                    global CUR_NODE = BB_nodes[1]
                    left_branching()
                    println("Proceed left Branching!")
                    left_branch += 1
                else #Means that we are going for the lower bound branching
                    global CUR_NODE = BB_nodes[1]
                    best_LB_branching()
                    println("Proceed Best LB branching!")
                    best_lb_branch += 1
                end
            end
        else
            break
        end

        if !GOOD_BRANCHINGS
            println("Branching issue")
            break
        end
        global RECORD_NOW = 1
        global ALREADY_OPTIMIZED = false
        feasible = true
    end

    if length(BB_nodes) > 0
        for leaves in BB_nodes
            push!(LB_leaves, LB_NODE[leaves])
        end
    end

    allocations = best_assign_it


    return best_assign_it, z, allocations, z_final, t_dp_total, t_cg_total, t_mp_total, best_lb_branch, TOT_NODE, LB_leaves, UB, prune_by_infeas, prune_by_bound
end