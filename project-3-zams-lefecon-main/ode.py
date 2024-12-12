def rk4(f,t,z,h,args=()):
    """
    <Description of routine goes here: what it does, how it's called>
    
    Arguments
        f(t,z,...)
            function that contains the RHS of the equation dz/dt = f(t,z,...)
    
        t : float
            The current time at which the function is evaluated.
            
        z : float or array
            The current value of the dependent variable at time t.
            
        h : float
            The step size we'll use to advance the solution from t to t + h.
    
        args (tuple, optional)
            additional arguments to pass to f
    
    Returns
        znew = z(t+h)
        This is the estimated value of z at time t + h, calculated using the RK4 method.
   """
    if not isinstance(args,tuple):
        args = (args,)
    
    k1 = h * f(t, z, *args)
    k2 = h * f(t + h/2, z + k1/2, *args)
    k3 = h * f(t + h/2, z + k2/2, *args)
    k4 = h * f(t + h, z + k3, *args)
    
    return z + (k1 + 2*k2 + 2*k3 + k4) / 6
