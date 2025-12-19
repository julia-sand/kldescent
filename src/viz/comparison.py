import os
import string

import numpy as np 
from sklearn.neighbors import KernelDensity
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

from ..refsolution.reference import RefSolution,GaussRefSolution

from .plots import *

plotter = PlotFuns() #initialise plotter

fig1,gs_cumulants = plotter.make_cumulant_fig_gridspec()
plotter.plot_free_diffusion_sol(ax,t_vec,cumulant)


self.make_cumulant_subplot(gs_cumulants[0, 0:2],"Position Mean",t_vec,cumulants_nn_plot,cumulants_ref_plot,test_samples)
self.make_cumulant_subplot(gs_cumulants[0, 2:4],"Position Variance",t_vec,cumulants_nn_plot,cumulants_ref_plot,test_samples)
self.make_cumulant_subplot(gs_cumulants[0, 4:],"Momentum Mean",t_vec,cumulants_nn_plot,cumulants_ref_plot,test_samples)
self.make_cumulant_subplot(gs_cumulants[1,:3],"Momentum Variance",t_vec,cumulants_nn_plot,cumulants_ref_plot,test_samples)
self.make_cumulant_subplot(gs_cumulants[1, 3:],"Cross Correlation",t_vec,cumulants_nn_plot,cumulants_ref_plot,test_samples)
                    