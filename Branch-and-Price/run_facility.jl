include("global_constants.jl")
ode_accuracy = :auto
#first element is relative tolerance, and second element is absolute tolerance
ode_accuracy_level = (1e-6, 1e-3)


output_master = []
# run lower bound branching only
lower_branching = true
cg_ip = false

G = ARGS[1]
F = parse(Int64, ARGS[2])
F_TOT = parse(Int64, ARGS[3])
FLEX = parse(Float64, ARGS[4])
DIST = parse(Float64, ARGS[5])
T = parse(Float64, ARGS[6])
D = parse(Float64, ARGS[7])

println("Group = ", G, " F = ", F, " F_TOT = ", F_TOT, " DIST = ", DIST)

TD = [[T, D, 1e-3, 0.01, G, F, F_TOT, DIST, FLEX]]

for values in TD
    global T = Int(values[1])
    global num_treatment_vals = Int(values[2])
    opt_gap = values[3]
    global batch_bound = values[4]
    global included_regions = [values[5]]
    global F = values[6]
    global F_TOT = values[7]
    global THRESHOLD_DISTANCE = values[8]
    global FLEXIBILITY = values[9]
    #global vaccine_budget = budget
    include("facilities.jl")
    include("main_facility.jl")
    t_val = @timed begin
        println("Going for group ", group, " with N = ", N_REGION, ", T = ", T, ", D = ", num_treatment_vals)
        global BATCHED_NORM_BOUND = batch_bound
        (state_space_sair, num_states, state_store_ss1, state_store_cost), t_state_space = @timed state_space(initial_state, "naive", ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level, batched = true);
        println("Number of states: ", num_states)
        output = []
        push!(output, batch_bound)
        push!(output, values[5])
        push!(output, T)
        push!(output, num_treatment_vals)
        push!(output, N_REGION)
        push!(output, F)
        push!(output, F_TOT)
        push!(output, THRESHOLD_DISTANCE)
        push!(output, FLEXIBILITY)
        push!(output, num_states)
        push!(output, t_state_space)
        # run BP
        reset_global_params()
        if cg_ip
            global C_CONSISTENT = []
            global C_REGION = []
            ini_columns(initial_state, state_space_sair, state_store_ss1, state_store_cost; ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            (model, z_final, C_CONSISTER, C_REGION, obj, IP, allocations_cg, time_dp_total, time_mp_total), t_cg_total = @timed column_generation(initial_state, state_space_sair, state_store_ss1, state_store_cost, C_CONSISTENT, C_REGION; ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            push!(output, time_mp_total)
            push!(output, time_dp_total)
            push!(output, t_cg_total)
            push!(output, nothing)
            push!(output, nothing)
            push!(output, nothing)
            push!(output, nothing)

        else
            global C_CONSISTENT = []
            global C_REGION = []
            worst, best = compute_best_and_worst(state_space_sair, state_store_ss1, state_store_cost)
            ini_columns(initial_state, state_space_sair, state_store_ss1, state_store_cost; ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            model, z_final, C_CONSISTENT, C_REGION, obj_cg, obj_cgip_ip, IP, z_ip, allocations_cg, max_time_total, time_mp_total, w, v, w_ip, v_ip = column_generation_with_ip_travel_round(initial_state, state_space_sair, state_store_ss1, state_store_cost, C_CONSISTENT, C_REGION, -1, -1; ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            reset_global_params()
            ini_columns(initial_state, state_space_sair, state_store_ss1, state_store_cost; ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            (model, z_final, C_CONSISTENT, C_REGION, obj_cg, obj_cgip, IP, z_ip, allocations_cg, max_time_total, time_mp_total, w, v, w_ip, v_ip), t_cg_total = @timed column_generation_with_ip_travel_round(initial_state, state_space_sair, state_store_ss1, state_store_cost, C_CONSISTENT, C_REGION, -1, 2; ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level)
            #upper_bound = cost_evaluation(allocations_cg)
            push!(output, time_mp_total)
            push!(output, max_time_total)
            push!(output, t_cg_total)
            upper_bound = sum(worst) - obj_cgip
            
            if lower_branching
                (best_assign_it_z, z_z, allocations_z, z_final_z, t_dp_total_z, t_cg_total_z, t_mp_total_z, best_lb_branch_z, nodes_z, LB_leaves_z, UBz, infeas, bound), t_bp_total  = 
                @timed branch_and_price_z_group_small(initial_state, state_space_sair, state_store_ss1, state_store_cost, true, -1, 2,
                ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level, upper_bound = upper_bound, optimality_gap = opt_gap);

                (best_assign_it_z, z_z, allocations_z, z_final_z, t_dp_total_z, t_cg_total_z, t_mp_total_z, best_lb_branch_z, nodes_z, LB_leaves_z, UBz, infeas, bound, branch_w, branch_x, branch_z, w, t_ip_total, UB_IP), t_bp_total  = 
                @timed branch_and_price_z_group(initial_state, state_space_sair, state_store_ss1, state_store_cost, true, -1, 2, 
                ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level, upper_bound = upper_bound, optimality_gap = opt_gap);
                
            else
                (final_decision, cg, plans, UBplot, LBplot, z, allocations, z_final, t_dp_total, t_cg_total, t_mp_total, left_branch, best_lb_branch), t_bp_total  = 
                @timed branch_and_price(initial_state, state_space_sair, state_store_ss1, state_store_cost, true, 
                ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level);
            end  
            #time for bi-partite branchings
            #push!(output, t_mp_total_x)
            #push!(output, t_dp_total_x)
            #push!(output, t_bp_x)

            #time for tri-partite branchings
            push!(output, t_mp_total_z)          
            push!(output, t_dp_total_z)
            push!(output, t_ip_total)
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
        push!(output, sum(worst) - obj_cg)
        push!(output, sum(worst) - obj_cgip)
        push!(output, sum(worst) - obj_cgip_ip)

        #push!(output, nodes_x)
        #push!(output, LB_leaves_x)
        #push!(output, UBip)

        push!(output, nodes_z)
        push!(output, LB_leaves_z)
        push!(output, UBz)
        push!(output, UB_IP)
        push!(output, infeas)
        push!(output, bound)

        push!(output, branch_w)
        push!(output, branch_x)
        push!(output, branch_z)
        push!(output, w)

    end 
    push!(output_master, output)
    output_master_df = DataFrame([getindex.(output_master, i) for i in 1:32], :auto, copycols=false)
    rename!(output_master_df, ["batched_norm", "Group","T","D","N", "F", "F_TOT", "Travel_Flex", "Cap_Flex","states","t_state","t_mp_root","t_dp_root","t_cgip", "t_mp_z", "t_dp_z", "t_ip_z","t_bp_z","t_total","lb_cg", "ub_cgh", "ub_cgip","nodes_z", "LB_z", "UB_H", "UB_IP","infeas", "bound", "branch_w", "branch_x", "branch_z", "F_chosen"])
    CSV.write("results_sensitivity/vaccinesfacilities_"*string(included_regions[1])*"_"*string(F)*"_"*string(F_TOT)*"_"*string(FLEX)*"_"*string(DIST)*"_"*string(T)*"_"*string(D)*".csv", output_master_df)
end
