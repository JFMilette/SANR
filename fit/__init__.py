"""
Fitting of a model.stack.Stack to measured reflectivity channels.

  problem   FitProblem: free parameters, data, model and cost (shared)
  de/       differential evolution: de_fit (run_de), de_window (GUI)
  dream/    Bayesian sampling with DREAM: dream (the sampler), dream_fit
            (posterior, sample, save / load), dream_plots (matplotlib),
            dream_window (GUI)
  panel     the Fit tool box of the Experimental tab (GUI)
"""
