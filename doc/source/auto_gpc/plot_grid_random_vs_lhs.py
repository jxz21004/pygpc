"""
Grid: Random vs LHS
===================

Choosing a sampling scheme
--------------------------

To calculate the coefficients of the gPC matrix, a number of random samples needs to be
picked to represent the propability space :math:`\\Theta` and enable descrete evaluations of the
polynomials. As for the computation of the coefficients, the input parameters :math:`\\mathbf{\\xi}`
can be sampled in a number of different ways. In **pygpc** the grid :math:`\\mathcal{G}` for this
application is constructed in `pygpc/Grid.py <../../../../pygpc/Grid.py>`_.

Random Sampling
^^^^^^^^^^^^^^^
In the case of random sampling the samples will be randomly from their Probability Density Function (PDF)
:math:`f(\\xi)`.

Latin Hypercube Sampling (LHS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To increase the information of each individual sampling point and to prevent undersampling, LHS is a simple
alternative to enhance the space-filling properties of the sampling scheme first established by
McKay et al. (2000).

.. [1] McKay, M. D., Beckman, R. J., & Conover, W. J. (2000). A comparison of three methods for selecting
   values of input variables in the analysis of output from a computer code. Technometrics, 42(1), 55-61.

To draw :math:`n` independent samples from a number of :math:`d`-dimensional parameters
a matrix :math:`\\Pi` is constructed with

.. math::

    \\pi_{ij} = \\frac{p_{ij} - u}{n}

where :math:`P` is a :math:`d \\times n` matrix of randomly perturbed integers
:math:`p_{ij} \\in \\mathbb{N}, {1,...,n}` and u is uniform random number :math:`u \\in [0,1]`.
"""

###############################################################################
# Constructing a simple LHS design
# --------------------------------
# We are going to create a simple LHS design for 2 random variables with 5 sampling points:
# sphinx_gallery_thumbnail_number = 3:

import pygpc
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# define parameters
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])

# define grid
lhs = pygpc.LHS(parameters_random=parameters, n_grid=0)

# draw samples
pi = lhs.get_lhs_grid(dim=2, n=25)

# plot
fig = plt.figure(figsize=(4, 4))
plt.scatter(pi[:,0], pi[:,1])
plt.xlabel("$x_1$", fontsize=12)
plt.ylabel("$x_2$", fontsize=12)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid()
plt.tight_layout()

#%%
# LHS Designs can further be improved upon, since the pseudo-random sampling procedure
# can lead to samples with high spurious correlation and the space filling capability
# in itself leaves room for improvement, some optimization criteria have been found to
# be adequate for compensating the initial designs shortcomings.
# 
# Optimization Criteria of LHS designs
# ------------------------------------
# Spearman Rank Correlation
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# For a sample size of :math:`n` the scores of each variable are converted to their Ranks :math:`rg_{X_i}`
# the Spearman Rank Correlation Coefficient is then the Pearson Correlation Coefficient applied to the rank
# variables :math:`rg_{X_i}`:
# 
# .. math::
#
#     r_s = \rho_{rg_{X_i}, rg_{X_j}} = \frac{cov(rg_{X_i}, rg_{X_j})}{\sigma_{rg_{X_i}} \sigma_{rg_{X_i}}}
#
# where :math:`\rho` is the pearson correlation coefficient, :math:`\sigma` is the standard deviation
# and :math:`cov` is the covariance of the rank variables
# 
# Maximum-Minimal-Distance
# ^^^^^^^^^^^^^^^^^^^^^^^^
# For creating a so called maximin distance design that maximizes the minimum inter-site distance, proposed by
# Johnson et al.
# 
# .. math::
#
#     \min_{1 \leqslant i, j \leqslant n, i \neq j} d(x_i,x_j),
# 
# where :math:`d` is the distance between two samples :math:`x_i` and :math:`x_j` and
# :math:`n` is the number of samples in a sample design.
# 
# .. math::
#
#     d(x_i,x_j) = d_ij = [ \sum_{k=1}^{m}|x_ik - x_jk| ^ t]^\frac{1}{t}, t \in {1,2}
#
# There is however a more elegant way of computing this optimization criterion as shown by Morris and Mitchell (1995),
# called the :math:`\varphi_P` criterion.
# 
# .. math::
#
#     \min\varphi_P \quad \text{subject to} \quad \varphi_P = [ \sum_{k = 1} ^ {s} J_id_i  ^ p]^\frac{1}{p},
# 
# where :math:`s` is the number of distinct distances, :math:`J` is an vector of indices of the distances
# and :math:`p` is an integer. With a very large :math:`p` this criterion is equivalent to the maximin criterion
#
# .. Morris, M. D. and Mitchell, T. J. ( (1995). Exploratory Designs for Computer Experiments.J. Statist. Plann.
#    Inference 43, 381-402.
# 
# LHS with enhanced stochastic evolutionary algorithm (ESE)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To achieve optimized designs with a more stable method and possibly quicker then by simply evaluating
# the criteria over a number of repetitions **pygpc** can use an ESE for achieving sufficient
# :math:`\varphi_P`-value. This algorithm is more appealing in its efficacy and proves to
# [sth about the resulting error or std in a low sample size].
# This method originated from Jin et al. (2005).
#
# .. Jin, R., Chen, W., Sudjianto, A. (2005). An efficient algorithm for constructing optimal
#    design of computer experiments. Journal of statistical planning and inference, 134(1), 268-287.

###############################################################################
# Comparison between a standard random grid and different LHS designs
# -------------------------------------------------------------------

from scipy.stats import spearmanr
import seaborn as sns

# define parameters
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])

# define grids for each criteria
lhs_basic = pygpc.LHS(parameters_random=parameters, n_grid=0)
lhs_corr = pygpc.LHS(parameters_random=parameters, n_grid=0)
lhs_maximin = pygpc.LHS(parameters_random=parameters, n_grid=0, options='maximin')
lhs_ese = pygpc.LHS(parameters_random=parameters, n_grid=0, options='ese')

# draw samples
dim = 5
n = 30
samples = []

samples.append(np.random.rand(n, dim))
samples.append(lhs_basic.get_lhs_grid(dim, n))
samples.append(lhs_corr.get_lhs_grid(dim, n, crit='corr'))
samples.append(lhs_maximin.get_lhs_grid(dim, n, crit='maximin'))
samples.append(lhs_ese.get_lhs_grid(dim, n, crit='ese'))

# calculate criteria
corrs = []
phis = []
name = []
variables = []

for i in range(5):
    corr = spearmanr(samples[i][:, 0], samples[i][:, 1])[0]
    corrs.append(corr)

for i in range(5):
    phip = lhs_basic.PhiP(samples[i])
    phis.append(phip)

variables.append(corrs)
name.append('corr')
variables.append(phis)
name.append('phi')

# plot results
fig = plt.figure(figsize=(16, 3))
titles = ['Random', 'LHS (standard)', 'LHS (corr opt)', 'LHS (Phi-P opt)', 'LHS (ESE)']

for i in range(5):
    text = name[0] + ' = {:0.2f} '.format(variables[0][i]) + "\n" + \
           name[1] + ' = {:0.2f}'.format(variables[1][i])
    plot_index = 151 + i
    plt.gcf().text((0.15 + i * 0.16), 0.08, text, fontsize=14)
    plt.subplot(plot_index)
    plt.scatter(samples[i][:, 0], samples[i][:, 1], color=sns.color_palette("bright", 5)[i])
    plt.title(titles[i])
    plt.gca().set_aspect('equal', adjustable='box')
plt.subplots_adjust(bottom=0.3)

#%%
# The initial LHS (standard) has already good space filling properties compared
# to the random sampling scheme (eg. less under sampled areas and less clustered areas,
# visually and quantitatively represented by the optimization criteria). The LHS (ESE)
# shows the best correlation and :math:`\varphi_P` criterion.

###############################################################################
# Convergence and stability comparison in gPC
# -------------------------------------------
# We are going to compare the different grids in a practical gPC example considering the Ishigami function.
# We are going to conduct gPC analysis for different approximation orders (grid sizes).
# Because we are working with random grids, we are interested in (i) the rate of convergence
# and (ii) the stability of the convergence. For that reason, we will repeat the analysis several times.
#
# Setting up the problem
# ^^^^^^^^^^^^^^^^^^^^^^
import pygpc
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

# grids to compare
grids = [pygpc.Random, pygpc.LHS, pygpc.LHS, pygpc.LHS, pygpc.LHS]
grids_options = [None, None, "corr", "maximin", "ese"]
grid_legend = ["Random", "LHS (standard)", "LHS (corr opt)", "LHS (Phi-P opt)", "LHS (ESE)"]
order = [2, 3, 4, 5, 6, 7, 8, 9, 10]
repetitions = 5

err = np.zeros((len(grids), len(order), repetitions))
n_grid = np.zeros(len(order))

# Model
model = pygpc.testfunctions.Ishigami()

# Problem
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x3"] = 0.
parameters["a"] = 7.
parameters["b"] = 0.1

problem = pygpc.Problem(model, parameters)

# gPC options
options = dict()
options["method"] = "reg"
options["solver"] = "Moore-Penrose"
options["interaction_order"] = problem.dim
options["order_max_norm"] = 1
options["n_cpu"] = 0
options["adaptive_sampling"] = False
options["gradient_enhanced"] = False
options["fn_results"] = None
options["error_type"] = "nrmsd"
options["error_norm"] = "relative"
options["matrix_ratio"] = 2
options["eps"] = 0.001
options["backend"] = "omp"

#%%
# Running the analysis
# ^^^^^^^^^^^^^^^^^^^^
for i_g, g in enumerate(grids):
    for i_o, o in enumerate(order):
        for i_n, n in enumerate(range(repetitions)):

            options["order"] = [o] * problem.dim
            options["order_max"] = o
            options["grid"] = g
            options["grid_options"] = grids_options[i_g]

            n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
                                                   order_glob_max=options["order_max"],
                                                   order_inter_max=options["interaction_order"],
                                                   dim=problem.dim)

            grid = g(parameters_random=problem.parameters_random,
                     n_grid=options["matrix_ratio"] * n_coeffs,
                     options=options["grid_options"])

            # define algorithm
            algorithm = pygpc.Static(problem=problem, options=options, grid=grid)

            # Initialize gPC Session
            session = pygpc.Session(algorithm=algorithm)

            # run gPC session
            session, coeffs, results = session.run()

            err[i_g, i_o, i_n] = pygpc.validate_gpc_mc(session=session,
                                                       coeffs=coeffs,
                                                       n_samples=int(1e4),
                                                       n_cpu=0,
                                                       output_idx=0,
                                                       fn_out=None,
                                                       plot=False)

        n_grid[i_o] = grid.n_grid

err_mean = np.mean(err, axis=2)
err_std = np.std(err, axis=2)

#%%
# Results
# ^^^^^^^
# Even after a small set of repetitions the :math:`\varphi_P` optimizing ESE will produce
# the best results regarding the aforementioned criteria, while also having less variation
# in its pseudo-random design. Thus is it possible to half the the root-mean-squared error
# :math:`\varepsilon` by using the ESE algorithm compared to completely random sampling the
# grid points, while also having a consistently small standard deviation.

fig, ax = plt.subplots(1, 2, figsize=[12,5])

for i in range(len(grids)):
    ax[0].errorbar(n_grid, err_mean[i, :], err_std[i, :], capsize=3, elinewidth=.5)
    ax[1].plot(n_grid, err_std[i, :])

for a in ax:
    a.legend(grid_legend)
    a.set_xlabel("$N_g$", fontsize=12)
    a.grid()

ax[0].set_ylabel("$\epsilon$", fontsize=12)
ax[1].set_ylabel("std($\epsilon$)", fontsize=12)

ax[0].set_title("gPC error vs original model (mean and std)")
_ = ax[1].set_title("gPC error vs original model (std)")
