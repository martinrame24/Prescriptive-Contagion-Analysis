include("global_constants.jl")

# :auto is "full accuracy", and :fixed is "approximate accuracy"
ode_accuracy = :auto
# first element is relative tolerance, and second element is absolute tolerance
ode_accuracy_level = (1e-6, 1e-3)
# create state space
output_master = []
# only run cg_ip without branching
cg_ip = false
# run lower bound branching only
lower_branching = true
batch_bound = 0.01

opt_gap = 1/1000
T = parse(Int64, ARGS[1])
Y = parse(Int64, ARGS[2])
X = parse(Int64, ARGS[3])
bound = parse(Float64, ARGS[4])


TD_values = [[T,Y,X,bound]]
for values in TD_values
    global T = Int(values[1])
    global D_x = Int(values[3])
    global K_y = Int(values[2])
    global batch_bound = values[4]
    #global vaccine_budget = budget
    include("social_media.jl")
    include("main_social.jl")
    t_val = @timed begin
        println("Going for T = ", T, ", X = ", D_x, ", Y = ", K_y)
        global BATCHED_NORM_BOUND = batch_bound
        (state_space_sair, num_states, state_store_ss1, state_store_cost), t_state_space = @timed state_space(initial_state, "naive", ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level, batched = true);
    
        output = []
        push!(output, batch_bound)
        push!(output, T)
        push!(output, D_x)
        push!(output, K_y)
        push!(output, N_REGION)
        push!(output, num_states)
        push!(output, t_state_space)
        # run BP
        reset_global_params()
        #This one we don't really care
        if cg_ip
            global C_BUDGET = []
            global C_REGION = []
            ini_columns(initial_state, state_space_sair, state_store_ss1, state_store_cost; ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            (model, z_final, C_BUDGET, C_REGION, obj, IP, allocations_cg, time_dp_total, time_mp_total), t_cg_total = @timed column_generation(initial_state, state_space_sair, state_store_ss1, state_store_cost, C_BUDGET, C_REGION; ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            push!(output, time_mp_total)
            push!(output, time_dp_total)
            push!(output, t_cg_total)
            push!(output, nothing)
            push!(output, nothing)
            push!(output, nothing)
            push!(output, nothing)

        #This is the one that matters
        else
            global C_BUDGET = []
            global C_REGION = []
            #Benchmarks costs
            worst = compute_worst(state_space_sair, state_store_ss1, state_store_cost)
            zero_plan = [[repeat([0], N_DECISION) for t in 1:T] for i in 1:N_REGION]
            real_cost_null = cost_evaluation(zero_plan)
            cost_no_plan_deaths = 0
            cost_uniform_plan_deaths = 0
            cost_region_cost_plan_deaths = 0
            cost_dynamic_plan_deaths = 0

            #CG
            ini_columns(initial_state, state_space_sair, state_store_ss1, state_store_cost; ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            (model, z_final, C_BUDGET, C_REGION, obj_cg, obj_cgip, IP, z_ip, allocations_cg, max_time_total, time_mp_total), t_cg_total = @timed column_generation_with_ip(initial_state, state_space_sair, state_store_ss1, state_store_cost, C_BUDGET, C_REGION; ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            reset_global_params()
            ini_columns(initial_state, state_space_sair, state_store_ss1, state_store_cost; ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            (model, z_final, C_BUDGET, C_REGION, obj_cg, obj_cgip, IP, z_ip, allocations_cg, max_time_total, time_mp_total), t_cg_total = @timed column_generation_with_ip(initial_state, state_space_sair, state_store_ss1, state_store_cost, C_BUDGET, C_REGION; ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            #upper_bound = cost_evaluation(allocations_cg)
            push!(output, time_mp_total)
            push!(output, max_time_total)
            push!(output, t_cg_total)
            upper_bound = sum(worst) - obj_cgip
            real_ub_cg = real_cost_null - cost_evaluation(allocations_cg)
            
            if lower_branching
                #Solve a fisrt small instance so that functions are already pre-compiled
                
                (best_assign_it, z, allocations, z_final, t_dp_total_x, t_cg_total_x, t_mp_total_x, best_lb_branch, nodes_x, LB_leaves_x, UBx), t_bp_x  =
                @timed branch_and_price_small(initial_state, state_space_sair, state_store_ss1, state_store_cost, true, 
                ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level, upper_bound = upper_bound, optimality_gap = opt_gap);
                (best_assign_it_z, z_z, allocations_z, z_final_z, t_dp_total_z, t_cg_total_z, t_mp_total_z, best_lb_branch_z, nodes_z, LB_leaves_z, UBz), t_bp_total  = 
                @timed branch_and_price_z_small(initial_state, state_space_sair, state_store_ss1, state_store_cost, true, 
                ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level, upper_bound = upper_bound, optimality_gap = opt_gap);
                
                #Real runs
                (best_assign_it, z, allocations, z_final, t_dp_total_x, t_cg_total_x, t_mp_total_x, best_lb_branch, nodes_x, LB_leaves_x, UBx), t_bp_x  =
                @timed branch_and_price(initial_state, state_space_sair, state_store_ss1, state_store_cost, true, 
                ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level, upper_bound = upper_bound, optimality_gap = opt_gap);

                (best_assign_it_z, z_z, allocations_z, z_final_z, t_dp_total_z, t_cg_total_z, t_mp_total_z, best_lb_branch_z, nodes_z, LB_leaves_z, UBz), t_bp_total  = 
                @timed branch_and_price_z(initial_state, state_space_sair, state_store_ss1, state_store_cost, true, 
                ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level, upper_bound = upper_bound, optimality_gap = opt_gap);
                real_ub_bp_x = real_cost_null - cost_evaluation(allocations)
                real_ub_bp_z = real_cost_null - cost_evaluation(allocations_z)
            #This one is useless too
            else
                (final_decision, cg, plans, UBplot, LBplot, z, allocations, z_final, t_dp_total, t_cg_total, t_mp_total, left_branch, best_lb_branch), t_bp_total  = 
                @timed branch_and_price(initial_state, state_space_sair, state_store_ss1, state_store_cost, true, 
                ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level);
            end  
            #time for bi-partite branchings
            push!(output, t_mp_total_x)
            push!(output, t_dp_total_x)
            push!(output, t_bp_x)

            #time for tri-partite branchings
            push!(output, t_mp_total_z)          
            push!(output, t_dp_total_z)
            push!(output, t_bp_total)
            
        end
    end
    
    push!(output, t_val.time)
    if cg_ip
        push!(output, nothing)
        push!(output, nothing)
        push!(output, cost_evaluation(allocations_cg))
        push!(output, nothing)
        push!(output, nothing)
        push!(output, nothing)
        push!(output, nothing)

    else
        push!(output, cost_no_plan_deaths)
        push!(output, cost_uniform_plan_deaths)
        push!(output, cost_region_cost_plan_deaths)
        push!(output, cost_dynamic_plan_deaths)

        #Costs for CG
        push!(output, sum(worst) - obj_cg)
        push!(output, sum(worst) - obj_cgip)
        push!(output, real_ub_cg)

        #Costs for bi-partite BP
        push!(output, nodes_x)
        push!(output, LB_leaves_x)
        push!(output, UBx)
        push!(output, real_ub_bp_x)

        #Costs for tri-partite BP
        push!(output, nodes_z)
        push!(output, LB_leaves_z)
        push!(output, UBz)
        push!(output, real_ub_bp_z)

    end 
    push!(output_master, output)
    output_master_df = DataFrame([getindex.(output_master, i) for i in 1:32], :auto, copycols=false)
    rename!(output_master_df, ["batched_norm","T","X", "Y","N","states","t_state","t_mp_root","t_dp_root","t_cgip", "t_mp_x", "t_dp_x", "t_bp_x", "t_mp_z", "t_dp_z", "t_bp_z","t_total", "no_plan_cost", "uniform_cost", "cost_based_cost", "dynamic_cost", "lb_cg", "ub_cgip", "real_cg", "nodes_x", "LB_x", "UB_x", "real_bp_x", "nodes_z", "LB_z", "UB_z", "real_bp_z"])

    CSV.write("results_social/social"*string(T)*"_"*string(X)*"_"*string(Y)*"_"*string(bound)*".csv", output_master_df)
end