# Group-Based Sparse Representation for Image Denoising

The problem formulation is:

![](./img/equation.png)

where in this cases, input y is a group of patches have non-local similarity within the whole image. The optimization is ADMM method.

The total flowchart is shown as:

![](./img/flowchart.png)

## Experiment

The noise image comes from mantual additive Gaussian distribution noise with zero-mean and 0.01 or 0.1 standard derivative.

![](./img/experiment.png)

