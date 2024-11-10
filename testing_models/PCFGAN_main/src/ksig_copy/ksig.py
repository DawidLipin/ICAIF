import cupy as cp
import numpy as np
import warnings

from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
from typing import Optional, Sequence, Tuple, Union
from numbers import Integral, Number

ArrayOnCPU = np.ndarray
ArrayOnGPU = cp.ndarray
ArrayOnCPUOrGPU = Union[cp.ndarray, np.ndarray]
_EPS = 1e-12

def check_positive_value(scalar: Number, name: str) -> Number:
  """Checks whether `scalar` is a positive number.

  Args:
    scalar: A variable to check.
    name: The name of the variable.

  Returns:
    The variable unchanged or raises an error if it is not positive.
  """
  if scalar <= 0:
    raise ValueError(f'The parameter \'{name}\' should have a positive value.')
  return scalar

def squared_norm(X: ArrayOnGPU, axis: int = -1) -> ArrayOnGPU:
  """Computes the squared norm by reducing over a given axis.

  Args:
    X: An n-dim. array to compute the norm of.
    axis: An axis to perform the reduction over.

  Returns:
    An (n-1)-dim. array containing the squared norms.
  """
  return cp.sum(cp.square(X), axis=axis)

def matrix_mult(X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None,
                transpose_X: bool = False, transpose_Y: bool = False
                ) -> ArrayOnGPU:
  """Performs batch matrix multiplication.

  Args:
    X: A batch of matrices.
    Y: Another batch of matrices (if not given uses `X`).
    transpose_X: Whether to transpose `X`.
    transpose_Y: Whether to transpose `Y`.

  Returns:
    The result of matrix multiplication, another batch of matrices.
  """
  subscript_X = '...ji' if transpose_X else '...ij'
  subscript_Y = '...kj' if transpose_Y else '...jk'
  return cp.einsum(
    f'{subscript_X},{subscript_Y}->...ik', X, Y if Y is not None else X)


def squared_euclid_dist(X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None
                        ) -> ArrayOnGPU:
  """Computes pairwise squared Euclidean distances.

  Args:
    X: An array of shape `[..., m, d]`.
    Y: Another array of shape `[..., n, d]`. Uses `X` if not given.

  Returns:
    An array of shape `[..., m, n]`.
  """
  X_n2 = squared_norm(X)
  if Y is None:
    D2 = (X_n2[..., :, None] + X_n2[..., None, :]
          - 2 * matrix_mult(X, X, transpose_Y=True))
  else:
    Y_n2 = squared_norm(Y, axis=-1)
    D2 = (X_n2[..., :, None] + Y_n2[..., None, :]
          - 2 * matrix_mult(X, Y, transpose_Y=True))
  return D2



class Kernel(BaseEstimator, metaclass=ABCMeta):
  """Base class for kernels.

  Deriving classes should implement the following methods:
    _K: Computes the kernel matrix between two data arrays.
    _Kdiag: Computes the diagonal kernel entries of a given data array.
    _validate_data: Performs any data checking and reshaping on data arrays.

  Warning: This class should not be used directly, use derived classes.
  """

  def fit(self, X: ArrayOnCPUOrGPU, y: Optional[ArrayOnCPUOrGPU] = None
            ) -> 'Kernel':
    return self

  @abstractmethod
  def _K(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
    """Computes the kernel matrix.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU (if not given uses `X`).

    Returns:
      The kernel matrix between `X` and `Y`, or `X` and `X` if `Y is None`.
    """

  @abstractmethod
  def _Kdiag(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes the diagonal kernel entries.

    Args:
      X: A data array on GPU.

    Returns:
      Diagonal entries of the kernel matrix of `X`.
    """

  @abstractmethod
  def _validate_data(X: ArrayOnCPUOrGPU, reset: bool = False
                     ) -> ArrayOnCPUOrGPU:
    """Validates the input data array `X`.

    This method returns `X` as derived classes might make changes to it.

    Args:
      X: A data array on CPU or GPU.
      reset: Whether to reset internal variables if any.

    Returns:
      Validated data on CPU or GPU.
    """

  def __call__(self, X: ArrayOnCPUOrGPU, Y: Optional[ArrayOnCPUOrGPU] = None,
               diag: bool = False, return_on_gpu: bool = False
               ) -> ArrayOnCPUOrGPU:
    """Implementes the basic call method of a kernel object.

    It takes as input one or two arrays, either on CPU (as `numpy`) or on
    GPU (as `cupy`), and computes the corresponding kernel matrix.

    Args:
      X: A data array on CPU or GPU.
      Y: An optional data array on CPU or GPU (if not given uses `X`).
      diag: Whether to compute only the diagonal. Ignores `Y` in this case.
      return_on_gpu: Whether to return the result on GPU.

    Returns:
      A kernel matrix or its diagonal entries on CPU or GPU.
    """
    # Validate data and move it to GPU.
    X = cp.asarray(self._validate_data(X))
    if diag:
      K = self._Kdiag(X)
    else:
      Y = cp.asarray(self._validate_data(Y)) if Y is not None else None
      K =  self._K(X, Y)
    if not return_on_gpu:
      K = cp.asnumpy(K)
    return K
  

class StaticKernel(Kernel, metaclass=ABCMeta):
  """Base class for static kernels.
  
  Note: Static kernels merge the last two axes for arrays with `ndim > 2`,
  so that it can be readily used on sequential data without pipeline changes.
  
  Deriving classes should implement the following methods:
    _K: Computes the kernel matrix between two data arrays.
    _Kdiag: Computes the diagonal kernel entries of a given data array.

  Warning: This class should not be used directly, use derived classes.
  """

  def _validate_data(self, X: ArrayOnCPUOrGPU, reset: bool = False
                     ) -> ArrayOnCPUOrGPU:
    """This method merges the last two axes for input arrays with `ndims > 2`.

    Args:
      X: A data array on CPU or GPU.
      reset: Provided for API consistency, unused.

    Returns:
      Reshaped data array on CPU or GPU.
    """
    # Merge the time axis with the feature axis.
    if X.ndim > 2:
      X = X.reshape(X.shape[:-2] + (-1,))
    return X
  

class StationaryKernel(StaticKernel):
  """Base class for stationary kernels.
  
  Stationary kernels considered here have diagonals equal to 1.
  
  Deriving classes should implement the following methods:
    _K: Computes the kernel matrix between two data arrays.

  Warning: This class should not be used directly, use derived classes.
  """

  def __init__(self, bandwidth: float = 1.) -> None:
    """Initializes the `StationaryKernel` object.

    Args:
      bandwidth: Bandwidth hyperparameter that inversely scales the input data.
    """
    self.bandwidth = check_positive_value(bandwidth, 'bandwidth')

  def _Kdiag(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes the diagonal entries for stationary kernel.

    Args:
      X: A data array on GPU.

    Returns:
      Diagonal entries of the kernel matrix.
    """
    return cp.full((X.shape[0],), 1)


class RBFKernel(StationaryKernel):
  """RBF kernel also called the Gaussian kernel ."""

  def _K(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
    """Computes the RBF kernel matrix.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU (if not given uses `X`).

    Returns:
      The kernel matrix between `X` and `Y`, or `X` and `X` if `Y is None`.
    """
    D2_scaled = squared_euclid_dist(X, Y) / cp.maximum(
      2*self.bandwidth**2, _EPS)
    return cp.exp(-D2_scaled)
  



class SignatureBase(Kernel, metaclass=ABCMeta):
  """Base class for signature kernels.

  Warning: This class should not be used directly, only derived classes.
  """

  def __init__(self, n_levels: int = 5, order: int = 1,
               difference: bool = True, normalize: bool = True,
               n_features: Optional[int] = None):
    """Initializer for `SignatureBase` base class.

    Args:
      n_levels: Number of signature levels.
      order: Signature embedding order.
      difference: Whether to take increments of lifted sequences in the RKHS.
      normalize: Whether to normalize kernel to unit norm in feature space.
      n_features: Optional, the number of features (state-space dim.).
        Provide this when feeding in flattened sequence arrays of ndim=2.

    Raises:
      ValueError: If `n_levels` is not positive.
      ValueError: If `n_features is not None` and it is not positive.
    """
    self.n_levels = check_positive_value(n_levels, 'n_levels')
    self.order = (self.n_levels if order <= 0 or order >= self.n_levels
                  else order)
    self.normalize = normalize
    self.difference = difference
    self.n_features = (check_positive_value(n_features, 'n_features')
                       if n_features is not None else None)

  def _validate_data(self, X: ArrayOnCPUOrGPU, reset: Optional[bool] = False
                     ) -> ArrayOnCPUOrGPU:
    """Validates the input data.

    Args:
      X: A data array on CPU or GPU.
      reset: Whether to reset already fitted parameters.

    Raises:
      ValueError: If the number of features in `X` != `n_features`.
    """

    n_features = (self.n_features_ if hasattr(self, 'n_features_')
                  and self.n_features_ is not None else self.n_features)
    if X.ndim == 2:
      if n_features is None or reset:
        warnings.warn(
          '`X` has` ndim==2. Assuming inputs are univariate time series.',
          'Recommend passing an `n_features` parameter during init when using',
          'flattened arrays of ndim==2.')
        n_features = 1
    elif X.ndim == 3:
      if n_features is None or reset:
        n_features = X.shape[-1]
      elif X.shape[-1] != n_features:
        raise ValueError(
          'Number of features in `X` does not match saved `n_features` param.')
    else:
      raise ValueError(
        'Only input sequence arrays with ndim==2 or ndim==3 are supported.')
    # Reshape data to ndim==3.
    X = X.reshape([X.shape[0], -1, n_features])
    if reset:
      self.n_features_ = n_features
    return X
  






def robust_sqrt(X: ArrayOnGPU) -> ArrayOnGPU:
  """Robust elementwise square root.

  Args:
      X: An array to take the elementwise square root of.

  Returns:
      An array of the same shape.
  """
  return cp.sqrt(cp.maximum(X, _EPS))


def matrix_diag(A: ArrayOnGPU) -> ArrayOnGPU:
  """Extracts the diagonals from a batch of matrices.

  Args:
    A: A batch of matrices of shape `[..., d, d]`.

  Returns:
    The extracted diagonals of shape `[..., d]`.
  """
  return cp.einsum('...ii->...i', A)





class SignatureKernel(SignatureBase):
  """Class for full-rank signature kernel."""

  def __init__(self, n_levels: int = 5, order: int = 1,
               difference: bool = True, normalize: bool = True,
               n_features: Optional[int] = None,
               static_kernel: Optional[StaticKernel] = None):
    """Initializes the `SignatureKernel` class.

    Args:
      n_levels: Number of signature levels.
      order: Signature embedding order.
      difference: Whether to take increments of lifted sequences in the RKHS.
      normalize: Whether to normalize kernel to unit norm in feature space.
      n_features: Optional, the number of features (state-space dim.).
        Provide this when feeding in flattened sequence arrays of ndim=2.
      static_kernel: A static kernel from `ksig.static.kernels`.

    Raises:
      ValueError: If `n_levels` is not positive.
      ValueError: If `n_features is not None` and it is not positive.
    """

    super().__init__(n_levels=n_levels, order=order, difference=difference,
                     normalize=normalize, n_features=n_features)
    self.static_kernel = static_kernel or RBFKernel()

  def _compute_embedding(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None,
                      diag: bool = False) -> ArrayOnGPU:
    """Computes the embedding of pairwise kernel evaluations.

    Args:
      X: An array of sequences on GPU.
      Y: Another array of sequences on GPU.
      diag: Whether to compute only the diagonals of K(X, X). Ignores `Y`.

    Returns:
      Pairwise static kernel evaluations required to compute the kernel.
    """
    if diag:
      M = self.static_kernel(X[..., None, :], return_on_gpu=True)
    else:
      if Y is None:
        M = self.static_kernel(X[:, None, :, None, :], X[None, :, :, None, :],
                               return_on_gpu=True)
      else:
        M = self.static_kernel(X[:, None, :, None, :], Y[None, :, :, None, :],
                               return_on_gpu=True)
    return M

  def _compute_kernel(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None,
                      diag: bool = False) -> ArrayOnGPU:
    """Computes the signature kernel matrix.

    Args:
      X: An array of sequences on GPU.
      Y: Another array of sequences on GPU.
      diag: Whether to compute only the diagonals of K(X, X). Ignores `Y`.

    Returns:
      Signature kernel matrix K(X, Y) or the diagonals of K(X, X).
    """
    # M has shape `[n_X, l_X, l_Y]` if `diag` else `[n_X, n_Y, l_X, l_Y]`.
    M = self._compute_embedding(X, Y, diag=diag)
    K = signature_kern(M, self.n_levels, order=self.order,
                       difference=self.difference,
                       return_levels=self.normalize)
    return K

  def _Kdiag(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes the diagonal kernel entries.

    Args:
      X: An array of sequences on GPU.

    Returns:
      Diagonal entries of the signature kernel matrix K(X, X).
    """
    if self.normalize:
      return cp.full((X.shape[0],), 1.)
    else:
      return self._compute_kernel(X, diag=True)

  def _K(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
    """Computes the kernel matrix.

    Args:
      X: An array of sequences on GPU.
      Y: Another array of sequences on GPU (if not given uses `X`).

    Returns:
      The kernel matrix between `X` and `Y`, or `X` and `X` if `Y is None`.
    """
    K = self._compute_kernel(X, Y)
    if self.normalize:
      if Y is None:
        K_Xd = utils.matrix_diag(K)
        if hasattr(self, 'is_log_space') and self.is_log_space:
          K -= 1./2 * (K_Xd[..., :, None] + K_Xd[..., None, :])
        else:
          K_Xd_sqrt = cp.maximum(robust_sqrt(K_Xd), _EPS)
          K /= K_Xd_sqrt[..., :, None] * K_Xd_sqrt[..., None, :]
      else:
        K_Xd = self._compute_kernel(X, diag=True)
        K_Yd = self._compute_kernel(Y, diag=True)
        if hasattr(self, 'is_log_space') and self.is_log_space:
          K -= 1./2 * (K_Xd[..., :, None] + K_Yd[..., None, :])
        else:
          K_Xd_sqrt = cp.maximum(robust_sqrt(K_Xd), _EPS)
          K_Yd_sqrt = cp.maximum(robust_sqrt(K_Yd), _EPS)
          K /= K_Xd_sqrt[..., :, None] * K_Yd_sqrt[..., None, :]
    # If log-space, then exponentiate now.
    if hasattr(self, 'is_log_space') and self.is_log_space:
      K = cp.exp(K)
    # If there is an `n_levels+1` axis in the beginning, do averaging.
    if K.ndim == 3:
      K = cp.mean(K, axis=0)
    return K