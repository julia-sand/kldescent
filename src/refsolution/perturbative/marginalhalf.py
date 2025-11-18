from .reference import *
#contains perturbative calculations for marginal halfbridge



def v11(t0,Tf):
    coeff = 1- np.exp(t0-Tf)
    return coeff*dqv00(t2,Tf)

def DU(t0):
    2*v11(t0,Tf)

if __name__=="__main__":
    boundary = UnderdampedLandauer()

    reference_ = RefSolution(boundary,folder_name=folder_name)