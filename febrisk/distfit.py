import scipy


class MLEDistFitter:
    """
    An abstract class to fit the distribution of a random variable x
    using the MLE method.
    
    Users should define the distribution of x in subclasses.
    """

    @staticmethod
    def ll(dist, x):
        """
        The log likelihood function.
        
        params:
            - dist: distribution of x, a callable object provided by scipy.stats
        """
        return dist.logpdf(x).sum()
    
    def dist(*args, **kwargs):
        """
        The distribution of x, which is to be determined.
        """
        raise NotImplementedError
    
    def fit(self, x, x0, constraints=None, **kwargs):
        """Estimate the parameters by maximizing log likelihood function."""
        
        # customize the log-likelihood function and negate it
        def negated_ll(args):
            """
            The negated log likelihood function, so that minimizing it achieves
            the same effect as maximizing the log likelihood function.
            
            args are passed to self.dist to generate the distribution of x
            """
            dist = self.dist(*args)
            return -self.ll(dist, x)

        all_constraints = getattr(self, 'constraints', {})
        if constraints:
            all_constraints.update(constraints)

        return scipy.optimize.minimize(negated_ll, x0=x0, constraints=all_constraints, **kwargs)
    
    
class TFitter(MLEDistFitter):
    """
    Fit the data using a t distribution.
    """
    constraints = ({"type": "ineq", "fun": lambda x: x[1]},  # the degree of freedom of t distribution is positive
                   {"type": "ineq", "fun": lambda x: x[2]})  # the scale parameter is also positive

    def dist(self, loc, df, scale):
        """
        Assume the data follows a T distribution
        whose degree of freedom is df and is scaled by scale.
        """
        return scipy.stats.t(loc=loc, df=df, scale=scale)

