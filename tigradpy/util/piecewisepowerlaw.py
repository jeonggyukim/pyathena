"""
Code stolen from ComoloPy
http://pydoc.net/CosmoloPy/0.1.104/cosmolopy.utils/
"""

import numpy as np

class PiecewisePowerlaw(object):
    """A piecewise powerlaw function.

    You can specify the intervals and power indices, and this class
    will figure out the coefficients needed to make the function
    continuous and normalized to unit integral.

    Notes
    -----

    Intervals are defined by an array l

    Powerlaw indicies by and array p

    a_n are the coefficients.
    
    f(x) = a_n x^{p_n} for l_{n-1} <= x < l_n

    Recursion relation for continuity:

    a_n = a_{n-1} l_n^{p_{n-1} - p_n}

    Integral of a piece:

    I_n = a_n p_n (l_{n+1}^{p_n - 1} - l_n^{p_n - 1})

    Total integral:

    I_tot = Sum_0^N I_n

    """

    def __init__(self, limits, powers,
                 coefficients=None,
                 externalval=0.0,
                 norm=True):
        """Defined a piecewise powerlaw.

        If coefficients is None then the coefficients are determined
        by requiring the function to be continuous and normalized to
        an integral of one.

        The function is composed of N powerlaws, where N = len(powers).

        len(limits) must be one greated than len(powers)

        Parameters
        ----------

        limits: array (length n+1)
            boundaries of the specified powerlaws. Must be one greater in
            length than coefficents and powers. Specify -numpy.infty for
            the first limit or numpy.infty for the last limit for
            unbounded powerlaws.

        coefficients: optional array (length n)
            values of the coefficient a_i

        powers: array (length n)
            values of the powerlaw indices p_i

        externalval: scalar
            Value to return outside the defined domain. None
            correspons to 'NaN'.

        norm: boolean
            Whether to normalize the integral of the function over the
            defined domain to unity.

        The resulting function takes a single, one-dimensional array of
        values on which to operate.

        """

        limits = np.atleast_1d(limits)
        powers = np.atleast_1d(powers)

        if not len(limits) == len(powers)+1:
            raise ValueError("limits must be one longer than powers.")

        if coefficients is None:
            coefficients = np.ones(len(powers))

            # Leaving a_0 = 1, apply the recurence relation.
            for n in range(1,len(powers)):
                coefficients[n] = (coefficients[n-1] *
                                   limits[n]**(powers[n-1] - powers[n]))
        else:
            coefficients = np.atleast_1d(coefficients)
            if not len(coefficients) == len(powers):
                raise ValueError("coefficients and powers must be"+
                                 " the same length.")

        # Find the integral of each piece.
        integrals = ((coefficients / (powers + 1.)) *
                     (limits[1:]**(powers + 1.) -
                      limits[:-1]**(powers + 1.)))
        if norm:
            # The total integral over the function.
            integralTot = np.sum(integrals)
            
            coefficients = coefficients / integralTot
            integrals = integrals /  integralTot

        for array in [limits, coefficients, powers]:
            if array.ndim > 1:
                raise ValueError("arguments must be a 1D arrays or scalars.")
        self._integrals = integrals
        self._limits = limits.reshape((-1,1))
        self._coefficients = coefficients.reshape((-1,1))
        self._powers = powers.reshape((-1,1))
        self._externalval = externalval
    
    def __call__(self, x):
        """Evaluate the powerlaw at values x.
        """
        x = np.atleast_1d(x)
        if x.ndim > 1:
            raise ValueError("argument must be a 1D array or scalar.")
        y = np.sum((self._coefficients * x**self._powers) *
                      (x >= self._limits[0:-1]) * (x < self._limits[1:]),
                      axis=0)

        y[x < self._limits[0]] = self._externalval
        y[x >= self._limits[-1]] = self._externalval
        return y

    def integrate(self, low, high, weight_power=None):
        """Integrate the function from low to high.

        Optionally weight the integral by x^weight_power.

        """
        limits = self._limits.flatten()
        coefficients = self._coefficients.flatten()
        powers = self._powers.flatten()
        if weight_power is not None:
            powers += weight_power
            # Integral of each piece over its domain.
            integrals = ((coefficients / (powers + 1.)) *
                         (limits[1:]**(powers + 1.) -
                          limits[:-1]**(powers + 1.)))
        else:
            integrals = self._integrals
        
        pairs = np.broadcast(low, high)
        integral = np.empty(pairs.shape)
        for (i, (x0, x1)) in enumerate(pairs):
            # Sort the integral limits.
            x0, x1 = list(np.sort([x0,x1]))

            # Select the pieces contained entirely in the interval.
            mask = np.logical_and(x0 < limits[:-1],
                                     x1 >= limits[1:]).flatten()
            indices = np.where(mask)
            if not np.any(mask):
                integral.flat[i] = 0

                # If the interval is outside the domain.
                if x0 > limits[-1] or x1 < limits[0]:
                    integral.flat[i] = 0
                    continue

                # Find out if any piece contains the entire interval:
                containedmask = np.logical_and(x0 >= limits[:-1],
                                                  x1 < limits[1:])
                # Three possibilites:
                if np.any(containedmask):
                    # The interval is contained in a single segment.
                    index = np.where(containedmask)[0][0]
                    integral.flat[i] = ((coefficients[index] /
                                         (powers[index] + 1.)) *
                                        (x1**(powers[index] + 1.) -
                                         x0**(powers[index] + 1.)))
                    continue
                elif x1 >= limits[0] and x1 < limits[1]:
                    # x1 is in the first segment.
                    highi = 0
                    lowi = -1
                elif x0 < limits[-1] and x0 >= limits[-2]:
                    # x0 is in the last segment:
                    lowi = len(limits) - 2
                    highi = len(limits)
                else:
                    # We must be spanning the division between a pair of pieces.
                    lowi = np.max(np.where(x0 >= limits[:-1]))
                    highi = np.min(np.where(x1 < limits[1:]))
                insideintegral = 0
            else:
                # Add up the integrals of the pieces totally inside the interval.
                insideintegral = np.sum(integrals[indices])

                lowi = np.min(indices) - 1
                highi = np.max(indices) + 1

            # Check that the integral limits are inside our domain.
            if x0 < limits[0] or lowi < 0:
                lowintegral = 0.
            else:
                lowintegral = ((coefficients[lowi] / (powers[lowi] + 1.)) *
                               (limits[lowi + 1]**(powers[lowi] + 1.) -
                                x0**(powers[lowi] + 1.)))
            if x1 > limits[-1] or highi > len(coefficients) - 1:
                highintegral = 0.
            else:
                highintegral = ((coefficients[highi] / (powers[highi] + 1.)) *
                                (x1**(powers[highi] + 1.) -
                                 limits[highi]**(powers[highi] + 1.)))
            integral.flat[i] = highintegral + insideintegral + lowintegral
        return integral
