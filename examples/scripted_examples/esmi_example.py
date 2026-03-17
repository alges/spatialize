"""Calculating spatial entropy for a synthetic sample."""
import numpy as np
import matplotlib.pyplot as plt
from spatialize.viz import plot_colormap_data, PlotStyle
from spatialize.gs.esmi import SpatialEntropy

# Prepare functions for generating synthetic scenario
def generate_regular_grid(ranges, step):
        """
        Generates regular grid.
        
        ranges: list of tuples (min, max) for each dimension
        step: spacing between points (same for all dimensions)
        
        Returns array of shape (n_points, n_dims)
        """
        # generate grid of points
        axes = [np.arange(low, high + step/2, step) for low, high in ranges]
        grids = np.meshgrid(*axes, indexing='ij')
        points = np.stack([g.ravel() for g in grids], axis=-1)

        # count decimal places
        s = str(step)
        if '.' in s:
            decimal_places = len(s.split('.')[-1])
        else:
            decimal_places = 0

        # round to decimal places of step
        return np.round(points, decimal_places)

def sample_spatial_normal(points, mu):
        """Sample values from normal distribution with mean "mu" and spatially dependent sigma given by the norm of the position.""" 
        std_x = np.linalg.norm(points, axis=-1)

        std = np.sqrt(std_x ** 2 + np.exp(-10) ** 2)
        return np.random.normal(mu, std) 

def generate_samples(n_samples, xi, mu):
        """Generates synthetic samples with a uniform spatial distribution.
        The value distribution of the samples is gaussian with mean "mu" and spatially dependent sigma given by the norm of the position.
        """
        min_values = xi.min(axis=0)
        max_values = xi.max(axis=0)
        
        # Base points
        n_dims = xi.shape[-1]
        points = np.random.random_sample((n_samples, n_dims))

        # Transform to uniform distributions within the ranges of the grid 
        for i, (a, b) in enumerate(list(zip(min_values, max_values))):
            points[:,i] = (b - a) * points[:,i] + a

        # Generate values 
        values = sample_spatial_normal(points, mu)
        
        return points, values

# Initialize synthetic scenario
np.random.seed(40)
grid_ranges =  [(-10, 10.01), (-10, 10.01)]
grid_step = 0.5
n_samples = 50
mu = 0

xi = generate_regular_grid(grid_ranges, grid_step)
points, values = generate_samples(n_samples, xi, mu)

# Calculate spatial entropy
entropy_estimator = SpatialEntropy(T=800, M=300, alpha_t=0.8, alpha_m=0.4, exponent=2.5)
estimates = entropy_estimator.calculate_entropy(points, values, xi)

# Visualize results
with PlotStyle(theme='publication', cmap='batlow') as style:
    fig, ax = plt.subplots(1,1, figsize=(5, 6), dpi=100)
    plot_colormap_data(estimates, ax=ax, xi_locations=xi,
                       griddata=False, cmap = style.cmap,
                       extent=[-10, 10.01, -10, 10.01])
    ax.scatter(points[:, 0], points[:, 1], s=8, c='magenta', label = 'samples')
    ax.legend(loc='upper left', bbox_to_anchor=(0.9, 1.1))
    ax.set_title('Differential Entropy Estimates')

plt.show()