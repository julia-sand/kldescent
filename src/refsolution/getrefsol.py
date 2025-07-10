import julia

j = julia.Julia(project="../swift_equilibrium")

#runs a julia script
#
j.include("test.jl")
integrate_half_gaussian(T,
                        epsilon,
                        sigma0,
                        sigmaT,
                        mu0,
                        muT,
                        outputloc)


if __name__=="__main__":
    pass