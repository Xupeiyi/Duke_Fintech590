import numpy as np
import scipy.stats
import scipy.optimize


class DistFitter:
    """
    An abstract class to fit the distribution of a random variable x
    using the MLE method.

    Users should define the distribution of x in subclasses.
    """
    constraints = tuple()

    def __init__(self, constraints=None):
        if constraints is not None:
            self.constraints = tuple([*self.constraints, *constraints])
        self.result = None

    @staticmethod
    def ll(dist, x):
        """
        The log likelihood function.

        params:
            - dist: distribution of x, a callable object provided by scipy.stats
        """
        return dist.logpdf(x).sum()

    def dist(self, *args, **kwargs):
        """
        The distribution of x, which is to be determined.
        """
        raise NotImplementedError

    def fit(self, x, x0, **kwargs):
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

        self.result = scipy.optimize.minimize(
            negated_ll, x0=x0, constraints=self.constraints, **kwargs
        )
    
    @property
    def fitted_dist(self):
        return self.dist(*self.result.x)

    @property
    def fitted_params(self):
        return tuple([*self.result.x])


class TFitter(DistFitter):
    """
    Fit a t distribution to the data.
    """

    # Inherent Constraints:
    # the degree of freedom of t distribution must be greater than 2
    # the scale parameter must be positive
    constraints = ({"type": "ineq", "fun": lambda x: x[1] - 2},
                   {"type": "ineq", "fun": lambda x: x[2]})

    def dist(self, loc, df, scale):
        return scipy.stats.t(loc=loc, df=df, scale=scale)

    def fit(self, x, x0=None, **kwargs):
        x0 = (np.mean(x), 2, np.std(x)) if x0 is None else x0
        return super().fit(x, x0=x0, **kwargs)


class NormalFitter(DistFitter):
    """
    Fit a normal distribution to the data.
    """

    # Inherent Constraints
    # the standard deviation must be positive
    constraints = ({"type": "ineq", "fun": lambda x: x[1]},)

    def dist(self, mean, std):
        return scipy.stats.norm(loc=mean, scale=std)

    def fit(self, x, x0=None, **kwargs):
        x0 = (np.mean(x), np.std(x)) if x0 is None else x0
        return super().fit(x, x0=x0, **kwargs)

