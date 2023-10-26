# unc_allegro_md

The project aims to develop a framework for uncertainty quantification on an accurate and data-efficient neural equivariant interatomic potential [Allegro](https://github.com/mir-group/allegro).

## Main idea

The project extract the edge feature vectors from a trained allegro model by passing through the training atomic frames, and fit a Gaussian mixture model (GMM) to capture the density distribution of these edge features. Then, we use the negative log-likelihood from the fitted GMM model as an uncertainty score. We then get the score on the testing edge features (obtained by passing through the testing atomic frames through the allegro model), and use the score to inform whether a testing atomic frame is in the training distribution or not.