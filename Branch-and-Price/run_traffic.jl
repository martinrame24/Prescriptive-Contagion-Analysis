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
batch_bounds = [0.01, 0.005, 0.001]

T = parse(Int64, ARGS[1])
X = parse(Int64, ARGS[2])
Y = parse(Int64, ARGS[3])
bound = parse(Float64, ARGS[4])
TD_values = [[T,X,Y,bound]]

for values in TD_values
    global T = Int(values[1])
    global ve = Int(values[2])
    global be = Int(values[3])
    global batch_bound = values[4]
    tol = 0.001
    #global vaccine_budget = budget
    include("traffic.jl")
    include("main.jl")
    t_val = @timed begin
        #println("Going for T = ", T, ", X = ", ve, ", Y = ", be)
        global BATCHED_NORM_BOUND = batch_bound
        println("Going for T = ", T, ", X = ", ve, ", Y = ", be, " bound = ", batch_bound)
        (state_space_sair, num_states, state_store_ss1, state_store_cost), t_state_space = @timed state_space(initial_state, "naive", ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level, batched = true);
    
        output = []
        push!(output, batch_bound)
        push!(output, T)
        push!(output, ve)
        push!(output, be)
        push!(output, N_REGION)
        push!(output, num_states)
        push!(output, t_state_space)
        # run BP
        reset_global_params()
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

        else
            global C_BUDGET = []
            global C_REGION = []

            #Compute Costs for benchmarks
            no_plan = [[(0,0) for t in 1:T] for i in 1:N_REGION]
            worst = compute_worst(state_space_sair, state_store_ss1, state_store_cost)
            uniform_plan = uniform()
            region_cost_plan = cost_based(initial_state)
            dynamic_plan = dynamic(initial_state)

            cost_no_plan_deaths = cost_evaluation(no_plan)
            cost_uniform_plan_deaths = cost_evaluation(uniform_plan)
            cost_region_cost_plan_deaths = cost_evaluation(region_cost_plan)
            cost_dynamic_plan_deaths = cost_evaluation(dynamic_plan)

            #We do standard stuff
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
            
            if lower_branching
                
                (best_assign_it, z, allocations, z_final, t_dp_total_x, t_cg_total_x, t_mp_total_x, best_lb_branch, nodes_x, LB_leaves_x, UBx), t_bp_x  =
                @timed branch_and_price(initial_state, state_space_sair, state_store_ss1, state_store_cost, true, 
                ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level, upper_bound = upper_bound, tolerance = tol);
                (best_assign_it_z, z_z, allocations_z, z_final_z, t_dp_total_z, t_cg_total_z, t_mp_total_z, best_lb_branch_z, nodes_z, LB_leaves_z, UBz), t_bp_total  = 
                @timed branch_and_price_z(initial_state, state_space_sair, state_store_ss1, state_store_cost, true, 
                ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level, upper_bound = upper_bound, tolerance = tol);
                

                (best_assign_it, z, allocations, z_final, t_dp_total_x, t_cg_total_x, t_mp_total_x, best_lb_branch, nodes_x, LB_leaves_x, UBx), t_bp_x  =
                @timed branch_and_price(initial_state, state_space_sair, state_store_ss1, state_store_cost, true, 
                ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level, upper_bound = upper_bound, tolerance = tol);
                #println("----------------------------------------------------------------------------")
                (best_assign_it_z, z_z, allocations_z, z_final_z, t_dp_total_z, t_cg_total_z, t_mp_total_z, best_lb_branch_z, nodes_z, LB_leaves_z, UBz), t_bp_total  = 
                @timed branch_and_price_z(initial_state, state_space_sair, state_store_ss1, state_store_cost, true, 
                ode_accuracy = ode_accuracy, ode_accuracy_level = ode_accuracy_level, upper_bound = upper_bound, tolerance = tol);

                cost_bp = cost_evaluation(allocations)
                cost_bp_z = cost_evaluation(allocations_z)
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
        push!(output, sum(worst))
        push!(output, cost_no_plan_deaths)
        push!(output, cost_uniform_plan_deaths)
        push!(output, cost_region_cost_plan_deaths)
        push!(output, cost_dynamic_plan_deaths)

        push!(output, sum(worst) - obj_cg)
        push!(output, sum(worst) - obj_cgip)

        push!(output, nodes_x)
        push!(output, LB_leaves_x)
        push!(output, UBx)
        push!(output, cost_bp)

        push!(output, nodes_z)
        push!(output, LB_leaves_z)
        push!(output, UBz)
        push!(output, allocations)
        push!(output, cost_bp_z)
        

    end 
    push!(output_master, output)
    output_master_df = DataFrame([getindex.(output_master, i) for i in 1:33], :auto, copycols=false)
    rename!(output_master_df, ["batched_norm","T","X", "Y","N","states","t_state","t_mp_root","t_dp_root","t_cgip", "t_mp_x", "t_dp_x", "t_bp_x", "t_mp_z", "t_dp_z", "t_bp_z","t_total", "worst_cost", "no_plan_cost", "uniform_cost", "cost_based_cost", "dynamic_cost", "lb_cg", "ub_cgip", "nodes_x", "LB_x", "UB_x", "Cost BPx","nodes_z", "LB_z", "UB_z", "Allocations", "Cost BP"])
    CSV.write("traffic/traffic_"*string(T)*"_"*string(ve)*"_"*string(be)*"_"*string(bound)*".csv", output_master_df)
end
