using DifferentialEquations;
using DataFrames;
using CSV;

function integrate_half_gaussian()
    
    #parameters
    T = 5
    epsilon = 0.2
    
    p = epsilon #[epsilon]

    sigma0 = 1
    sigmaREF = 0.25^2
    sigp_ref = 1

    pos_mu_end = 0
    mom_mu_end = 0
    
    #boundary conds
    pos_var_start = sigma0
    x_var_start = 0 
    mom_var_start = 1

    function v11_coeff(sigma_pos_final,sigma_mom_final,sigma_cc_final)
        return (1/sigmaREF) - (sigma_mom_final/(sigma_pos_final*sigma_mom_final-sigma_cc_final^2))
    end

    function v12_coeff(sigma_pos_final,sigma_mom_final,sigma_cc_final)
        return (sigma_cc_final/(sigma_mom_final*sigma_pos_final-sigma_cc_final^2))
    end
    
    function v22_coeff(sigma_pos_final,sigma_mom_final,sigma_cc_final)
        return 1 - (sigma_pos_final/(sigma_pos_final*sigma_mom_final-sigma_cc_final^2))
    end

    vp_end = 0
        
    tspan = (0.0,T)

    #this is the function for the pde we are solving
    function varianceevolution!(du, u, p, t)
        v11, v12, v22, c11, c12, c22 = u
        epsilon = p
        du[1] = 2*(v12^2) - 2*((c12*v22/c11)^2)
        du[2] = v12 + 2*v22*v12 -epsilon*v11 + 2*c12*(v22^2)/c11
        du[3] = 2.0*(v22-v12*epsilon) ##
        du[4] = 2.0*epsilon*c12
        du[5] = epsilon*c22 - 2*c11*v12 - c12 - 2*v22*c12 #-epsilon*(-c22) + 2*c11*v12 - c12*(1-2*v22)
        du[6] = 2 - 2*c22 - 4*c12*v12 - ((4*(c12^2)*v22)/(c11))
        
    end

    #the boundary condition function 
    function varbc!(residual, u, p, t)
        residual[1] = u[1][4] - pos_var_start #the solution at the start
        residual[2] = u[1][5] - x_var_start
        residual[3] = u[1][6] - mom_var_start
        residual[4] = u[end][1] - v11_coeff(u[end][4],u[end][6],u[end][5])#Vqq_end(u[end][4],u[end][5]) #Vqq_end #the solution at the end
        residual[5] = u[end][2] - v12_coeff(u[end][4],u[end][6],u[end][5])#Vqp_end(u[end][6],u[end][4],u[end][5])
        residual[6] = u[end][3] - v22_coeff(u[end][4],u[end][6],u[end][5])#Vpp_end(u[end][6],u[end][5])
    end

    #initial guess
    u0_var = [0.0,0.0,0.0,1.0,0.0,1.0]

    bvp1 = BVProblem(varianceevolution!, varbc!, u0_var, tspan, p);

    sol1 = solve(bvp1,LobattoIIIa5(),dt=0.1)

    ##SAVE CSV HERE
    df = DataFrame(sol1)

    rename!(df, [:t, :Vqq, :Vqp, :Vpp, :Qt, :Ct, :Pt]) #rename 
    CSV.write("variance_half.csv",df)
    
    println("variance csv ready.")

    
    function u2(v22,v12,c12,c11)
        return 2*v12 + 2*(c12*v22)/c11
    end
    function u1(v2,s1,s2,v22,c12,c11)
        return 0#2*v2 + 2*s2*v22 - 2*(c12*s1*v22)/c11
    end

    #df = df_var
    df[!, "U2"] = u2.(df[!,"Vpp"], df[!,"Vqp"], df[!,"Ct"], df[!,"Qt"])
    df[!, "u1"] = u1.(0,0,0, df[!,"Vpp"], df[!,"Ct"], df[!,"Qt"])
    CSV.write(string("src/refsolution/UnderdampedHalfNet/refsol_v2.csv"),df)
    println("completed.")
        
end

     
integrate_half_gaussian()