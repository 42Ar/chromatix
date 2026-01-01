from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from chromatix.utils.utils import toarray
from einops import rearrange
from jax import Array


__all__ = [
    "_broadcast_1d_to_channels",
    "_broadcast_1d_to_polarization",
    "_broadcast_1d_to_innermost_batch",
    "_broadcast_1d_to_grid",
    "_broadcast_2d_to_grid",
    "_squeeze_grid_to_3d",
    "_broadcast_2d_to_spatial",
    "_check_shape_and_expand_dims",
    "_broadcast_scalar_and_expand_dims"
]


Tensor = Union[Array | np.ndarray]


def _broadcast_1d_to_channels(x: ArrayLike, ndim: int) -> Tensor:
    """Broadcast 1D arrays of of size `C` to `(B... H W C [1 | 3])`.
    Scalars are interpreted as 1D arrays of length 1."""
    x = toarray(x)
    x = x.__array_namespace__().atleast_1d(x)
    shape_spec = "c -> " + ("1 " * (ndim - 2)) + "c 1"
    return rearrange(x, shape_spec)


def _broadcast_1d_to_polarization(x: ArrayLike, ndim: int) -> Tensor:
    """Broadcast 1D arrays of size `P` to `(B... H W C [1 | 3])`.
    Scalars are interpreted as 1D arrays of length 1."""
    x = toarray(x)
    x = x.__array_namespace__().atleast_1d(x)
    shape_spec = "p -> " + ("1 " * (ndim - 1)) + "p"
    return rearrange(x, shape_spec)


def _broadcast_1d_to_innermost_batch(x: ArrayLike, ndim: int) -> Tensor:
    """Broadcast 1D array of size `B` to left of `(H W)` in `(B... H W C [1 | 3])`.
    Scalars are interpreted as 1D arrays of length 1."""
    x = toarray(x)
    x = x.__array_namespace__().atleast_1d(x)
    shape_spec = "b ->" + " 1" * (ndim - 5) + " b 1 1 1 1"
    return rearrange(x, shape_spec)


def _broadcast_1d_to_grid(x: ArrayLike, ndim: int) -> Tensor:
    """Broadcast 1D array of size `2` to `(2 B... H W C 1)`.
    Scalars are interpreted as 1D arrays of length 1."""
    x = toarray(x)
    x = x.__array_namespace__().atleast_1d(x)
    shape_spec = "d ->" + "d" + " 1" * (ndim - 4) + " 1 1 1 1"
    return rearrange(x, shape_spec, d=2)


def _broadcast_2d_to_grid(x: Tensor, ndim: int) -> Tensor:
    """Broadcast 2D array of shape `2 C` to `(2 B... H W C [1 | 3])`.
    Useful for vectorial ops on grids."""
    shape_spec = "d c ->" + "d" + " 1" * (ndim - 4) + " 1 1 c 1"
    return rearrange(x, shape_spec, d=2)


def _squeeze_grid_to_3d(x: Tensor, ndim: int) -> Tensor:
    """Squeeze array of shape `(2 B... H W C [1 | 3])` to 3D array of shape `(2 C 1)`.
    Useful for vectorial ops on grids."""
    shape_spec = "d" + " 1" * (ndim - 4) + " 1 1 c 1 -> d c 1"
    return rearrange(x, shape_spec, d=2)


def _broadcast_2d_to_spatial(x: Tensor, ndim: int) -> Tensor:
    """Broadcast 2D array of shape `(H W)` to `(B... H W C [1 | 3])`.
    If the array already has the specified number of dimensions, it
    is returned immediately."""
    if x.ndim != ndim:
        shape_spec = "h w ->" + ("1 " * (ndim - 4)) + "h w 1 1"
        return rearrange(x, shape_spec)
    return x


def _check_shape_and_expand_dims(x: ArrayLike, dims: int, shape: tuple[int, ...]) -> Array | np.ndarray:
    """Raises a ``ValueError`` if the shape of ``x`` does not start with ``shape``.
    Returns an array of a rank equal to ``dims`` by appending axes of size 1. If
    the provided array has a rank exceeding ``dims`` a ``ValueError`` is raised."""
    _x = toarray(x)
    if _x.shape[:len(shape)] != shape:
        raise ValueError(f"array has invalid shape: must start with {shape} but has {_x.shape}")
    if _x.ndim > dims:
        raise ValueError("provided array has to many axes")
    return _x[..., *[None]*(dims - _x.ndim)]


def _broadcast_scalar_and_expand_dims(x: ArrayLike, dims: int, n: int) -> Array | np.ndarray:
    """Convert ``x`` to an array of shape ``(n, ...)`` with a rank of ``dims`` by
    appending axes of size 1. Scalars are broadcasted; arrays must have size ``n``
    of the first dimension. If the array has an invalid shape a ValueError is raised."""
    _x = toarray(x)
    xp = _x.__array_namespace__()
    if _x.ndim == 0:
        _x = xp.full(n, _x)
    return _check_shape_and_expand_dims(_x, dims, (n,))
