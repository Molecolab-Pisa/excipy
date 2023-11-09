from functools import partial
from collections import defaultdict
from collections.abc import Iterable
import numpy as np
import scipy.linalg as scipy_la
from .database import type_in_database, get_params, DatabaseError
from .util import pbar, squared_distances

# ===================================================================
# Some functions to compute kernels
# just the ones we need, there is no general infrastructure here
#


def matern52_kernel(x1, x2, lengthscale, variance):
    dist2 = 5.0 * squared_distances(x1=x1, x2=x2) / (lengthscale**2)
    dist = np.sqrt(np.maximum(dist2, 1e-300))
    return variance * (1.0 + dist + (1.0 / 3.0) * dist2) * np.exp(-dist)


#
# Site energy prediction in vacuum
#

# since computing k_mm is expensive and, once done, is ok for
# all other frames of the same molecule type (e.g., CLA), we keep
# a cache of its cholesky decomposition L, k_mm = L @ L.T, here
_PREDICT_SITE_ENERGIES_VAC_CACHE = {}


def predict_site_energies_chlorophyll_vac(encoding, model_params, residue_id, chl):
    # compute matern kernel, m = n. train points, n = n. new points
    k_nm = matern52_kernel(
        x1=encoding,
        x2=model_params["x_train"],
        lengthscale=model_params["lengthscale"],
        variance=model_params["variance"],
    )
    # predict the mean
    mean = np.dot(k_nm, model_params["coefs"]) + model_params["mu"]

    # predict the variance
    cache_key = (chl, "L_m")
    if cache_key in _PREDICT_SITE_ENERGIES_VAC_CACHE:
        L_m = _PREDICT_SITE_ENERGIES_VAC_CACHE[cache_key]

    else:
        k_mm = matern52_kernel(
            x1=model_params["x_train"],
            x2=model_params["x_train"],
            lengthscale=model_params["lengthscale"],
            variance=model_params["variance"],
        ) + (model_params["sigma"] ** 2 + 1e-8) * np.eye(
            model_params["x_train"].shape[0]
        )
        L_m = scipy_la.cholesky(k_mm, lower=True)

        # update cache
        _PREDICT_SITE_ENERGIES_VAC_CACHE[cache_key] = L_m

    G_mn = scipy_la.solve_triangular(L_m, k_nm.T, lower=True)

    variance = matern52_kernel(
        encoding,
        encoding,
        lengthscale=model_params["lengthscale"],
        variance=model_params["variance"],
    )

    variance = variance - np.dot(G_mn.T, G_mn)

    return mean, variance


# this function is the same for chl b
predict_site_energies_cla_vac = partial(
    predict_site_energies_chlorophyll_vac, chl="CLA"
)
predict_site_energies_chl_vac = partial(
    predict_site_energies_chlorophyll_vac, chl="CHL"
)

_PREDICT_SITE_ENERGIES_FUNCS = {
    ("CLA", "VAC"): predict_site_energies_cla_vac,
    ("CHL", "VAC"): predict_site_energies_chl_vac,
}

#
# wrapper
#


def predict_site_energies(encoding, type, model_params, residue_id, kind):
    """
    Predict the site energy of each molecule with Gaussian Process Regression.
    Arguments
    ---------
    encodings   : list of ndarray, (num_samples, num_features)
                Molecule encodings
    types       : list of str
                List of molecule types (used to fetch the model)
    residue_ids : list of str
                List of Residue IDs
    kind        : str
                Kind of site energy to predict (vac, env)
    Returns
    -------
    site energies : defaultdict
                  Dictionary with mean and variance of each
                  prediction.
    """

    iterator = pbar(
        encoding,
        total=encoding.shape[0],
        desc=": Predicting siten",
        ncols=79,
    )

    siten = defaultdict(list)

    func = _PREDICT_SITE_ENERGIES_FUNCS[(type.upper(), kind.upper())]

    for x in iterator:
        # shape (1, n_features)
        x = np.atleast_2d(x)

        mean, variance = func(x, model_params, residue_id)

        siten["y_mean"].append(mean)
        siten["y_var"].append(variance)

    for key in siten.keys():
        siten[key] = np.concatenate(siten[key], axis=0)

    return siten


def predict_tresp_charges(encodings, descriptors, types, residue_ids):
    """
    Predict the TrEsp charges of each molecule.
    Arguments
    ---------
    encodings    : list of ndarray, (num_samples, num_features)
                 Molecule encodings
    descriptors  : list of objects
                 Descriptors
    types        : list of str
                 List of molecule types
    residue_ids  : list of str
                 List of Residue IDs
    Returns
    -------
    tresp_charges : list of ndarray, (num_samples, num_atoms)
                  Predicted TrEsp charges
    """
    iterator = pbar(
        zip(encodings, descriptors, types, residue_ids),
        total=len(encodings),
        desc=": Predicting TrEsp",
        ncols=79,
    )
    tresp_charges = []
    for encoding, descriptor, type, residue_id in iterator:
        iterator.set_postfix(residue_id=f"{residue_id}")
        # Instantiate a regressor
        # The alpha parameter here is unimportant, as the regression
        # parameters are loaded from the database
        ridge = LinearRidgeRegression()
        ridge.load_params(type)
        # Note that the predicted charges have to be permuted
        # again in order to match the correct atom ordering
        y_permuted = ridge.predict(encoding)
        inverse_permutation = descriptor.permutator.inverse_transform
        y = inverse_permutation(y_permuted)
        tresp_charges.append(y)
    return tresp_charges


class LinearRidgeRegression(object):
    """
    Linear Ridge Regression
    This implementation solves the linear problem through the manipulation
    of the Gram matrix K = XX^T. Leave-One-Out Cross Validation (LOO-CV) for
    selecting the best regularization parameter is supported.
    This implementation is inspired by the scikit-learn one. The theory is
    taken from here:
    http://cbcl.mit.edu/publications/ps/MIT-CSAIL-TR-2007-025.pdf
    """

    def __init__(self, alpha=1.0):
        """
        Arguments
        ---------
        alpha   : float, iterable of floats
                regularization parameter(s)
        """
        self.alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        iterable_alpha = isinstance(value, Iterable)
        # check whether alpha is given as a single value or as an iterable.
        # if given as a single value, cross-validation is disabled.
        # if given as an iterable, cross-validation is enabled and used to
        # select the best alpha value.
        if iterable_alpha:
            self.cross_validate = True
        else:
            self.cross_validate = False
        # ensure that alpha is positive
        if iterable_alpha:
            positive = np.all(np.asarray(value) > 0)
        else:
            positive = value > 0
        if not positive:
            raise RuntimeError("Parameter `alpha` must be positive.")
        self._alpha = value

    def _preprocess_input(self, X, Y):
        # Store the means of X and Y
        # mean(Y) is used as the intercept when making predictions
        # mean(X) is subtracted from X, as we work with a centered matrix
        self.X_mean_ = np.mean(X, axis=0)
        self.Y_mean_ = np.mean(Y, axis=0)
        X_centered = X - self.X_mean_
        return X_centered, Y

    @property
    def params(self):
        if not hasattr(self, "X_mean_"):
            return {}
        if not hasattr(self, "Y_mean_"):
            return {}
        if not hasattr(self, "w_"):
            return {}
        return {
            "X_mean": self.X_mean_.tolist(),
            "Y_mean": self.Y_mean_.tolist(),
            "w": self.w_.tolist(),
        }

    def load_params(self, type):
        """
        Load the parameters for a molecule of type
        `type` from the database.
        """
        if type_in_database(type):
            params = get_params(type)
            for key, val in params.items():
                params[key] = np.asarray(val)
            self.w_ = params["w"]
            self.X_mean_ = params["X_mean"]
            self.Y_mean_ = params["Y_mean"]
        else:
            raise DatabaseError(f"No parameters for molecule type {type}")
        return self

    @staticmethod
    def _compute_gram_matrix(X):
        """
        Compute the Gram matrix of X, K=XX^T
        """
        return np.dot(X, X.T)

    @staticmethod
    def _compute_regression_coefficients(X, c):
        """
        Compute the regression coefficients `w`.
        Arguments
        ---------
        X        : ndarray, (num_samples, num_features)
                 input matrix
        c        : ndarray, (num_samples, num_outputs)
                 dual coefficients
        """
        return np.dot(X.T, c)

    @staticmethod
    def _decay_factor(
        eigenvalues, alpha, penalize_intercept=True, num_samples=None, Q=None
    ):
        """
        Computes the "decay" factor (Lambda + a)^-1, where Lambda is the list
        of eigenvalues of K, and a is the regularization parameter.
        A correction is applied in order to not penalize the intercept. Note
        that this correction should be applied only if the gram matrix K is
        computed from an input matrix X augmented with a column of ones.
        Arguments
        ---------
        eigenvalues        : ndarray, (num_samples,)
                           eigenvalues of the gram matrix
        alpha              : float
                           regularization parameter
        penalize_intercept : bool
                           whether to penalize the intercept or not
        num_samples        : int
                           number of samples
        Q                  : ndarray, (num_samples, num_samples)
                           eigenvectors of the gram matrix
        """
        decay = 1.0 / (eigenvalues + alpha)
        # If the intercept is penalized, we return the decay
        # factor as is. Note that if the kernel matrix is computed
        # from a X not augmented with ones, the intercept is not
        # implicitely included and this function should be run with
        # `penalize_intercept=True`.
        if penalize_intercept:
            return decay
        # If the intercept is not to be penalized, then we have to
        # zero the entry of the decay factor that corresponds to the
        # intercept.
        # This is equivalent to what is done in scikit-learn, see:
        # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py
        else:
            # We assume that if `penalize_intercept` is False, then also
            # `num_samples` and `Q` are provided.
            unit_vector = np.ones(num_samples) / num_samples
            alignment = np.abs(np.dot(unit_vector, Q))
            intercept_index = np.argmax(alignment)
            decay[intercept_index] = 0.0
            return decay

    @staticmethod
    def _compute_dual_coefficients(decay, Q, QtY):
        """
        Compute the dual coefficients efficiently given the decay factor, the
        eigenvectors Q of the kernel matrix, and the dot product QtY of Q^T and
        the target Y.
        Arguments
        ---------
        decay   : ndarray, (num_samples,)
                decay factor
        Q       : ndarray, (num_samples, num_samples)
                eigenvectors of the gram matrix
        QtY     : ndarray, (num_samples, num_outputs)
                dot product Q^T.Y between the eigenvectors and the target
        """
        # running the multiplication with an additional "fake" dimension
        # is faster than computing the dot product. This trick is
        # taken from scikit-learn's _ridge.py.
        # diag((Lambda + aI)^-1).Q^T.Y
        decayQtY = decay[:, None] * QtY
        # dual coefficients
        # Q.diag((Lambda+aI)^-1).Q^T.Y
        return np.dot(Q, decayQtY)

    @staticmethod
    def _compute_diagonal_inverse_G(decay, Q):
        """
        Computes the diagonal of the inverse of G(a)=K+aI, given
        the decay factor and the eigenvectors Q of the kernel matrix.
        Arguments
        ---------
        decay   : ndarray, (num_samples,)
                the decay factor
        Q       : ndarray, (num_samples, num_samples)
                eigenvectors of the gram matrix
        """
        # trick also taken from scikit-learn's _ridge.py.
        # a fake dimension is also added at the end to support
        # broadcasting when Y is 2D.
        return np.sum(decay * Q**2, axis=-1)[:, None]

    @staticmethod
    def _mean_squared_error_loocv(c, diag_G_inv):
        """
        Computes the mean-squared LOO-CV error using the efficient
        result available for Ridge regression.
        Arguments
        ---------
        c          : ndarray, (num_samples, num_outputs)
                   dual coefficients
        diag_G_inv : ndarray, (num_samples,1)
                   diagonal of G^-1
        """
        loo_error = c / diag_G_inv
        return np.mean(loo_error**2)

    def _fit_direct(self, X, Y):
        """
        Solve the linear system G(a)c=Y directly, with
        G(a)=K+aI and where 'a' is the regularization parameter.
        The linear system is solved with Cholesky.
        This method is called only when `alpha` is given as a single number.
        Arguments
        ---------
        X       : ndarray, (num_samples, num_features)
                input matrix
        Y       : ndarray, (num_samples, num_outputs)
                target matrix
        """
        num_samples, num_features = X.shape
        K = self._compute_gram_matrix(X)
        G = K + self.alpha * np.eye(num_samples)
        c = scipy_la.solve(G, Y, assume_a="pos")
        self.w_ = self._compute_regression_coefficients(X, c)

    def _fit_loocv(self, X, Y):
        """
        Solve the linear system G(a)c=Y, with G(a)=K+aI
        and where 'a' is the regularization parameter.
        The linear system is solved using the eigendecomposition of the gram matrix K.
        This method is called only when `alpha` is given as a list of numbers.
        Arguments
        ---------
        X       : ndarray, (num_samples, num_features)
                input matrix
        Y       : ndarray, (num_samples, num_outputs)
                target matrix
        """
        num_samples, num_features = X.shape
        # Note that we add a 1 here in order to be able to
        # fit also the intercept when computing the error using
        # the ridge's LOO-CV formula, c / diag(G). If the intercept
        # is not accounted for, the LOO error is wrong.
        # Adding the 1 is equivalent to computing the gram matrix K
        # from the input matrix X with an additional column of ones.
        K = self._compute_gram_matrix(X) + 1.0
        evals, Q = scipy_la.eigh(K)
        QtY = np.dot(Q.T, Y)
        # Initialize the mean squared error of LOO-CV
        # and the best dual coefficients.
        self.mse_loocv_ = np.infty
        best_c = None
        for alpha in self.alpha:
            decay = self._decay_factor(
                evals, alpha, penalize_intercept=False, num_samples=num_samples, Q=Q
            )
            c = self._compute_dual_coefficients(decay, Q, QtY)
            diag_G_inv = self._compute_diagonal_inverse_G(decay, Q)
            error = self._mean_squared_error_loocv(c, diag_G_inv)
            if error < self.mse_loocv_:
                best_c = c
                self.best_alpha_ = alpha
                self.mse_loocv_ = error
        self.w_ = self._compute_regression_coefficients(X, best_c)

    def fit(self, X, Y):
        """
        Fit the regressor.
        If `alpha` is provided as a list of values, Leave-One-Out Cross Validation
        is used to set the best value of `alpha`.
        Arguments
        ---------
        X        : ndarray, (num_samples, num_features)
                 input matrix
        Y        : ndarray, (num_samples, num_outputs)
                 target matrix
        """
        X = X.copy()
        X, Y = self._preprocess_input(X, Y)
        if self.cross_validate:
            self._fit_loocv(X, Y)
        else:
            self._fit_direct(X, Y)
        return self

    def predict(self, X):
        """
        Predict that target values `Yhat` of `X`.
        Arguments
        ---------
        X        : ndarray, (num_samples, num_features)
                 input matrix
        """
        return self.Y_mean_ + np.dot(X - self.X_mean_, self.w_)
