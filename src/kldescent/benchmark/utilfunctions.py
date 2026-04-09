"""Useful functions used in computing perturbative solution"""

import os
import csv

import numpy as np

def apply_filter(y,filter_delta=None):
    """convolve array with box filter"""
    
    if filter_delta is None:
        #default to 5pc of the data vector
        filter_delta = int(np.floor(len(y)*0.05))

    #apply convolution with moving averages
    out = np.convolve(y,(1/filter_delta)*np.ones(filter_delta),'same')
    
    return out

def append_array_to_csv(filename,data):
    """append numpy array to a csv"""
    
    with open(filename,"a") as file:
      np.savetxt(file,data,fmt="%10.6f",delimiter=',')
  
def interpolate_(q,xs,phis):
    """Interpolate function phi from data points xs to q"""

    idx = np.argsort(xs)

    return np.interp(q,xs[idx],phis[idx],left=0,right=0)

def normalise(x,y):
    idx = np.argsort(x)
    return y.copy()/np.trapezoid(y[idx],x[idx])

def zchop(x,tol=1e-10):
   
    return np.where(np.abs(x)<tol,0,x)

def make_csv_and_header(save_location,filename,header):
    #create the directory
    filename_full = save_location+"/"+filename+".csv"
    
    with open(filename_full,"w") as file: 
        writer = csv.writer(file,delimiter=" ",lineterminator="\n")
        writer.writerow(header)
    
    return filename_full

def save_array(arr_out,save_location,filename,header):
    """saves an array as a csv to the given location
    args: 
        arr : ndarray
            array to save
        save_location : str
            filepath relative to where the code is run
        filename : str
            name of the file
        header : str 
            comment to add at the top of the csv
    """

    #make dir if does not exist
    os.makedirs(save_location,exist_ok=True)
    np.savetxt(save_location+"/"+filename+".csv", arr_out ,fmt="%10.6f",header = header,delimiter=',')
    return

def TVRegDiff(data, itern, alph, u0=None, scale='small', ep=1e-6, dx=None,
              precondflag=True,
              diffkernel='abs', cgtol=1e-4, cgmaxit=100):
    """
    Estimate derivatives from noisy data based using the Total 
    Variation Regularized Numerical Differentiation (TVDiff) 
    algorithm.

    Parameters
    ----------
    data : ndarray
        One-dimensional array containing series data to be
        differentiated.
    itern : int
        Number of iterations to run the main loop.  A stopping
        condition based on the norm of the gradient vector g
        below would be an easy modification.  No default value.
    alph : float    
        Regularization parameter.  This is the main parameter
        to fiddle with.  Start by varying by orders of
        magnitude until reasonable results are obtained.  A
        value to the nearest power of 10 is usally adequate.
        No default value.  Higher values increase
        regularization strenght and improve conditioning.
    u0 : ndarray, optional
        Initialization of the iteration.  Default value is the
        naive derivative (without scaling), of appropriate
        length (this being different for the two methods).
        Although the solution is theoretically independent of
        the initialization, a poor choice can exacerbate
        conditioning issues when the linear system is solved.
    scale : {large' or 'small' (case insensitive)}, str, optional   
        Default is 'small'.  'small' has somewhat better boundary
        behavior, but becomes unwieldly for data larger than
        1000 entries or so.  'large' has simpler numerics but
        is more efficient for large-scale problems.  'large' is
        more readily modified for higher-order derivatives,
        since the implicit differentiation matrix is square.
    ep : float, optional 
        Parameter for avoiding division by zero.  Default value
        is 1e-6.  Results should not be very sensitive to the
        value.  Larger values improve conditioning and
        therefore speed, while smaller values give more
        accurate results with sharper jumps.
    dx : float, optional    
        Grid spacing, used in the definition of the derivative
        operators.  Default is the reciprocal of the data size.
    plotflag : bool, optional
        Flag whether to display plot at each iteration.
        Default is True.  Useful, but adds significant
        running time.
    precondflag: bool, optional
        Flag whether to use a preconditioner for conjugate gradient solution.
        Default is True. While in principle it should speed things up, 
        sometimes the preconditioner can cause convergence problems instead,
        and should be turned off. Note that this mostly makes sense for 'small'
        scale problems; for 'large' ones, the improved preconditioner is one
        of the main features of the algorithms and turning it off defeats the
        point.
    diffkernel: str, optional
        Kernel to use in the integral to smooth the derivative. By default it's
        the absolute value, |u'| (value: "abs"). However, it can be changed to
        being the square, (u')^2 (value: "sq"). The latter produces smoother
        derivatives, whereas the absolute values tends to make them more blocky.
        Default is abs.
    cgtol: float, optional
        Tolerance to use in conjugate gradient optimisation. Default is 1e-4.
    cgmaxit: int, optional
        Maximum number of iterations to use in conjugate gradient optimisation. 
        Default is 100


    Returns
    -------
    u : ndarray
        Estimate of the regularized derivative of data.  Due to
        different grid assumptions, length(u) = length(data) + 1
        if scale = 'small', otherwise length(u) = length(data).
    """

    # Make sure we have a column vector
    data = np.array(data)
    assert len(data.shape) == 1, "data is not one-dimensional"
    # Get the data size.
    n = len(data)

    # Default checking. (u0 is done separately within each method.)
    if dx is None:
        dx = 1.0 / n

    # Different methods for small- and large-scale problems.
    if (scale.lower() == 'small'):

        # Differentiation operator
        d0 = -np.ones(n)/dx
        du = np.ones(n-1)/dx
        dl = np.zeros(n-1)
        dl[-1] = d0[-1]
        d0[-1] *= -1

        D = sparse.diags([dl, d0, du], [-1, 0, 1])
        DT = D.transpose()

        # Antidifferentiation and its adjoint
        def A(x): return (np.cumsum(x) - 0.5 * (x + x[0])) * dx

        def AT(x): return np.concatenate([[sum(x[1:])/2.0],
                                          (sum(x)-np.cumsum(x)+0.5*x)[1:]])*dx

        # Default initialization is naive derivative

        if u0 is None:
            u0 = D*data

        u = u0.copy()
        # Since Au( 0 ) = 0, we need to adjust.
        ofst = data[0]
        # Precompute.
        ATb = AT(ofst - data)        # input: size n

        # Main loop.
        for ii in range(1, itern+1):
            if diffkernel == 'abs':
                # Diagonal matrix of weights, for linearizing E-L equation.
                Q = sparse.spdiags(1. / (np.sqrt((D * u)**2 + ep)), 0, n, n)
                # Linearized diffusion matrix, also approximation of Hessian.
                L = dx * DT * Q * D
            elif diffkernel == 'sq':
                L = dx * DT * D
            else:
                raise ValueError('Invalid diffkernel value')

            # Gradient of functional.
            g = AT(A(u)) + ATb + alph * L * u

            # Prepare to solve linear equation.
            if precondflag:
                # Simple preconditioner.
                P = alph * sparse.spdiags(L.diagonal() + 1, 0, n, n)
            else:
                P = None

            def linop(v): return (alph * L * v + AT(A(v)))
            linop = splin.LinearOperator((n, n), linop)

            s, info_i = sparse.linalg.cg(
                linop, g, x0=None, tol=cgtol, maxiter=cgmaxit,
                callback=None, M=P, atol='legacy')

            # Update solution.
            u = u - s
            # Display plot.
            if plotflag:
                plt.plot(u)
                plt.show()

    elif (scale.lower() == 'large'):

        # Construct anti-differentiation operator and its adjoint.
        def A(v): return np.cumsum(v)

        def AT(w): return (sum(w) * np.ones(len(w)) -
                           np.transpose(np.concatenate(([0.0],
                                                        np.cumsum(w[:-1])))))
        # Construct differentiation matrix.
        c = np.ones(n)
        D = sparse.spdiags([-c, c], [0, 1], n, n) / dx
        mask = np.ones((n, n))
        mask[-1, -1] = 0.0
        D = sparse.dia_matrix(D.multiply(mask))
        DT = D.transpose()
        # Since Au( 0 ) = 0, we need to adjust.
        data = data - data[0]
        # Default initialization is naive derivative.
        if u0 is None:
            u0 = np.concatenate(([0], np.diff(data)))
        u = u0
        # Precompute.
        ATd = AT(data)

        # Main loop.
        for ii in range(1, itern + 1):

            if diffkernel == 'abs':
                # Diagonal matrix of weights, for linearizing E-L equation.
                Q = sparse.spdiags(1. / (np.sqrt((D * u)**2 + ep)), 0, n, n)
                # Linearized diffusion matrix, also approximation of Hessian.
                L = DT * Q * D
            elif diffkernel == 'sq':
                L = DT * D
            else:
                raise ValueError('Invalid diffkernel value')

            # Gradient of functional.
            g = AT(A(u)) - ATd
            g = g + alph * L * u
            # Build preconditioner.
            if precondflag:
                c = np.cumsum(range(n, 0, -1))
                B = alph * L + sparse.spdiags(c[::-1], 0, n, n)
                # droptol = 1.0e-2
                R = sparse.dia_matrix(np.linalg.cholesky(B.todense()))
                P = np.dot(R.transpose(), R)
            else:
                P = None
            # Prepare to solve linear equation.

            def linop(v): return (alph * L * v + AT(A(v)))
            linop = splin.LinearOperator((n, n), linop)

            s, info_i = sparse.linalg.cg(
                linop, -g, x0=None, tol=cgtol, maxiter=cgmaxit, callback=None,
                M=P, atol='legacy')
            
            # Update current solution
            u = u + s

        u = u / dx

    return u
