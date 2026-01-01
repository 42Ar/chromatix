import dataclasses
import operator
from typing import Self, override, TypeVar
from jax import Array
from jax._src.lax.control_flow.solves import _check_shapes
from jax.typing import ArrayLike as JaxArrayLike
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from chex import assert_rank
from chromatix.utils.shapes import (
    _broadcast_1d_to_channels,
    _broadcast_1d_to_grid,
    _broadcast_2d_to_grid,
    _check_shape_and_expand_dims,
    _broadcast_scalar_and_expand_dims
)
from chromatix.utils import create_grid, toarray
import equinox as eqx


SubclassesRaster = TypeVar("SubclassesRaster", bound="Raster")


class Raster(eqx.Module):
    """
    Represents a quantity that is sampled on an evenly spaced grid of
    size ``(H, W)``. 
    
    The quantity is stored in the array ``u`` of shape ``(B... H W C...)``
    where ``B...`` are called batch axes and ``C...`` are called channel
    axes. The array ``_dx`` is of shape ``(2, D...)`` and stores the
    sample spacing, where ``_dx[0]`` is the spacing in y direction
    (corresponds to ``H``) and ``_dx[1]`` is the spacing in x-direction
    (corresponds to ``W``). The shape of ``C...`` must be equal to ``D...``
    except that some axes can have length 1 in ``D...`` even if the
    corresponding axes in ``C...`` has not length 1. This allows for
    simplified broadcasting. 

    Notes
    -----
    The array ``_dx`` can also be a numpy array, to enable certain
    precomputations (like Pupil masks) and assertions during compile time
    of jax. 
    """

    u: Array  # (B... H W C...)
    _dx: Array | np.ndarray  # (2 D...)

    @classmethod
    def _normalize_dx(cls, dx: ArrayLike, channels_dims: int) -> Array | np.ndarray:
        return _broadcast_scalar_and_expand_dims(dx, channels_dims + 1, 2)

    def _check_shapes(self):
        """Raises an error if invalid or inconsistent shapes are detected."""
        if self.ndim < 2:
            raise ValueError("cannot construct a Raster object with less than two dimensions")
        if self.n_batch_dims < 0:
            raise ValueError("attempted to construct a raster with invalid "
                    f"number of axes in u or _dx (u.shape={self.u.shape}, _dx.shape={self._dx.shape})")
        if self._dx.shape[0] != 2:
            raise ValueError("the first axis of _dx must have size 2, but _dx has "
                    f"shape {self._dx.shape}")
        if not all((sc == 1 or sc == su) for sc, su in zip(self._dx.shape[1:], self.channel_shape)):
            raise ValueError("the channel shape of _dx is incompatible with the channel "
                    f"shape of u (u.shape={self.u.shape}, _dx.shape={self._dx.shape})")

    def __post_init__(self):
        self._check_shapes()

    def replace(self: SubclassesRaster, **changes) -> SubclassesRaster:
        result = dataclasses.replace(self, **changes)
        result._check_shapes()
        return result
    
    @classmethod
    def create_raster(cls, u: Array, dx: ArrayLike, channels_dims: int = 0) -> Self:
        return cls(u, cls._normalize_dx(dx, channels_dims))

    @property
    def ndim(self) -> int:
        """Total number of axes of the data array"""
        return self.u.ndim
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the complex field."""
        return self.u.shape

    @property
    def n_channel_dims(self) -> int:
        """Number of channel axes"""
        return self._dx.ndim - 1

    @property
    def channel_shape(self) -> tuple[int, ...]:
        """Shape of the batch axes"""
        return self.shape[self.ndim - self.n_channel_dims:]
    
    @property
    def n_batch_dims(self) -> int:
        """Number of batch axes"""
        return self.ndim - self._dx.ndim - 1

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Shape of the batch axes"""
        return self.shape[:self.n_batch_dims]

    @property
    def spatial_dims(self) -> tuple[int, int]:
        """Indices of the height (y) and width (x) axes of self.u"""
        return (self.n_batch_dims, self.n_batch_dims + 1)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Returns the spatial size as a tuple ``(height, width)``"""
        return (self.u.shape[self.spatial_dims[0]], self.u.shape[self.spatial_dims[1]])

    def __neg__(self):
        return self.replace(u=-self.u)

    def _binary_op(self: SubclassesRaster, operator, other: JaxArrayLike | "Raster", reverse: bool) -> SubclassesRaster:
        if isinstance(other, Raster):
            other = other.u
        elif not isinstance(other, (np.ndarray, Array, np.bool_, np.number, bool, int, float, complex)):
            return NotImplemented
        if reverse:
            res = operator(other, self.u)
        else:
            res = operator(self.u, other)            
        return self.replace(u=res)
        
    def __add__(self: SubclassesRaster, other: JaxArrayLike | "Raster") -> SubclassesRaster:
        return self._binary_op(operator.add, other, False)

    def __radd__(self: SubclassesRaster, other: JaxArrayLike) -> SubclassesRaster:
        return self._binary_op(operator.add, other, True)

    def __sub__(self: SubclassesRaster, other: JaxArrayLike | "Raster") -> SubclassesRaster:
        return self._binary_op(operator.sub, other, False)

    def __rsub__(self: SubclassesRaster, other: JaxArrayLike) -> SubclassesRaster:
        return self._binary_op(operator.sub, other, True)

    def __mul__(self: SubclassesRaster, other: JaxArrayLike | "Raster") -> SubclassesRaster:
        return self._binary_op(operator.mul, other, False)

    def __rmul__(self: SubclassesRaster, other: JaxArrayLike) -> SubclassesRaster:
        return self._binary_op(operator.mul, other, True)

    def __truediv__(self: SubclassesRaster, other: JaxArrayLike | "Raster") -> SubclassesRaster:
        return self._binary_op(operator.truediv, other, False)

    def __rtruediv__(self: SubclassesRaster, other: JaxArrayLike) -> SubclassesRaster:
        return self._binary_op(operator.truediv, other, True)

    def __floordiv__(self: SubclassesRaster, other: JaxArrayLike | "Raster") -> SubclassesRaster:
        return self._binary_op(operator.floordiv, other, False)

    def __rfloordiv__(self: SubclassesRaster, other: JaxArrayLike) -> SubclassesRaster:
        return self._binary_op(operator.floordiv, other, True)

    def __mod__(self: SubclassesRaster, other: JaxArrayLike | "Raster") -> SubclassesRaster:
        return self._binary_op(operator.mod, other, False)

    def __rmod__(self: SubclassesRaster, other: JaxArrayLike) -> SubclassesRaster:
        return self._binary_op(operator.mod, other, True)

    def __pow__(self: SubclassesRaster, other: JaxArrayLike | "Raster") -> SubclassesRaster:
        return self._binary_op(operator.pow, other, False)

    def __rpow__(self: SubclassesRaster, other: JaxArrayLike) -> SubclassesRaster:
        return self._binary_op(operator.pow, other, True)

    @property
    def conj(self: SubclassesRaster) -> SubclassesRaster:
        return self.replace(u=jnp.conj(self.u))

    @property
    def dx(self) -> Array | np.ndarray:
        """
        Returns the sample spacing as a numpy array of shape
        ``(2, B..., 1, 1, D...)``, where ones are inserted for
        the batch axes.
        """
        return self._dx[:, *[None]*(self.n_batch_dims + 2)]

    @property
    def _dk(self) -> Array | np.ndarray:
        """
        Analog of ``_dx`` in Fourier space, the returned array has shape ``(2, D...)``.
        """
        xp = self._dx.__array_namespace__()
        return 1.0/(self._dx*xp.array(self.spatial_shape)[:, *[None]*self.n_channel_dims])

    @property
    def dk(self) -> Array | np.ndarray:
        """
        Returns the k-space sample spacing as a numpy array
        of shape ``(2, B..., 1, 1, D...)``, where ones are
        inserted for the batch axes.
        """
        return self._dk[:, *[None]*(self.n_batch_dims + 2)]

    @property
    def grid(self) -> Array:
        """
        The grid of pixel coordinates as an array of shape ``(2 B... H W D...)``,
        where ones are inserted for the batch axes. The 2 entries along the first
        dimension represent the y and x grids, respectively. The returned grid
        is Fourier centered.
        """
        return create_grid(self.spatial_shape, spacing=self._dx)[:, *[None]*self.n_batch_dims]

    @property
    def k_grid(self) -> Array:
        """
        The frequency grid (without factor 2*pi) as an array of shape ``(2 B... H W D...)``,
        where ones are inserted for the batch axes. The 2 entries along the first dimension
        represent the k_y and k_x grids, respectively. The returned grid is Fourier centered.
        """
        return create_grid(self.spatial_shape, spacing=self._dk)[:, *[None]*self.n_batch_dims]


class Field(Raster):
    """
    Container describing a chromatic light field at a 2D plane.

    The shape of a ``Field`` object is ``(B..., H, W, C, [1 | 3])``, where
    ``B...`` denotes an arbitrary number of batch dimensions, ``H`` and ``W``
    are the height and width, and ``C`` is the channel dimension corresponding
    to different wavelengths in the spectrum. The final dimension has size
    either 1 for the scalar approximation (``ScalarField``) or 3 for the full
    vectorial case (``VectorField``). Any function in Chromatix that operates on
    ``Field`` objects can work with both ``ScalarField`` and ``VectorField``
    instances unless otherwise stated.

    To ensure correct broadcasting behavior, attributes that could otherwise
    be one-dimensional arrays are stored with additional singleton dimensions.

    For convenience, the class methods ``ScalarField.create()`` and
    ``VectorField.create()`` are provided.

    Attributes
    ----------
    _spectrum : jax.Array or np.ndarray
        Array of shape ``(C,)`` or ``(1,)`` that stores the wavelengths of
        the field.
    _spectral_density : jax.Array or np.ndarray
        Weights associated with the wavelength in ``_spectrum``. Stored as
        an array of shape ``(C,)`` or ``(1,)``.
    _origin : jax.Array or np.ndarray
        Defines a shift of the sampling grid relative to the Fourier origin.
        Stored as an array of shape ``(2, C)`` or ``(2, 1)``.
    """

    _spectrum: Array | np.ndarray  # (C) or (1,)
    _spectral_density: Array | np.ndarray # (C,) or (1,)
    _origin: Array | np.ndarray  # (2, C) or (2, 1)

    @classmethod
    def _normalize_spec_array(cls, spec: ArrayLike) -> Array | np.ndarray:
        _spec = toarray(spec)
        xp = _spec.__array_namespace__()
        _spec = xp.atleast_1d(_spec)
        if _spec.ndim != 1:
            raise ValueError("spectrum and spectral_density must be 1d arrays or scalars")
        return _spec
    
    @classmethod
    def _normalize_origin(cls, origin: ArrayLike) -> Array | np.ndarray:
        return _check_shape_and_expand_dims(origin, 2, (2,))

    @classmethod
    def empty_like(
        cls,
        field: Self,
        dx: ArrayLike | None = None,
        shape: tuple[int, int] | None = None,
        spectrum: ArrayLike | None = None,
        spectral_density: ArrayLike | None = None,
        origin: ArrayLike | None = None,
    ) -> Self:
        """
        Copy over attributes of ``field`` to a new ``Field`` object, with the
        option of changing some attributes.

        Note that this function overwrites the field `u` with a new empty field
        of same dtype, but with batch axes set to a size of 1.
        """
        if dx is None:
            _dx = field._dx
        else:
            _dx = cls._normalize_dx(dx, 2)
        if shape is None:
            shape = field.spatial_shape
        else:
            assert len(shape) == 2
        if spectrum is None:
            _spectrum = field._spectrum
        else:
            _spectrum = cls._normalize_spec_array(spectrum)
        if spectral_density is None:
            _spectral_density = field._spectral_density
        else:
            _spectral_density = cls._normalize_spec_array(spectral_density)
        if origin is None:
            _origin = field._origin
        else:
            _origin = cls._normalize_origin(origin)
        u = jnp.empty(
                (*[1]*field.n_batch_dims, *shape, *field.u.shape[-2:]), dtype=field.u.dtype)
        return cls(u, _dx, _spectrum, _spectral_density, _origin)

    @classmethod
    def _create(
        cls,
        vector_axis_size: int,
        dx: ArrayLike,
        spectrum: ArrayLike,
        spectral_density: ArrayLike,
        u: ArrayLike | None = None,
        shape: tuple[int, int] | None = None,
        origin: ArrayLike | None = None,
    ) -> Self:
        """Refer to docstrings of callers."""
        _dx = cls._normalize_dx(dx, 2)
        _spectrum = cls._normalize_spec_array(spectrum)
        _spectral_density = cls._normalize_spec_array(spectral_density)
        _spectral_density = _spectral_density / _spectral_density.sum()
        if u is None:
            if shape is None or len(shape) != 2:
                raise ValueError("must specify shape as a length 2 tuple if u is None")
            u = jnp.zeros((1, *shape, len(_spectrum), vector_axis_size), dtype=jnp.complex64)
        else:
            u = jnp.array(u)
        if origin is None:
            _origin = np.zeros((2, 1))
        else:
            _origin = cls._normalize_origin(origin)
        # shape checking is implicitly done when the object is constructed
        return cls(u, _dx, _spectrum, _spectral_density, _origin)

    @override
    def _check_shapes(self):
        super()._check_shapes()
        if self.u.ndim < 4:
            raise ValueError("field must be an array with at least 4 dimensions: (B... H W C 1)")
        if self._dx.shape not in [(2, 1, 1), (2, self.u.shape[-2], 1)]:
            raise ValueError(f"_dx has invalid shape")
        if self._spectrum.shape not in [(1,), (self.u.shape[-2])]:
            raise ValueError(f"_spectrum has invalid shape")
        if self._spectral_density.shape not in [(1,), (self.u.shape[-2])]:
            raise ValueError(f"_spectrum has invalid shape")
        if self._origin.shape not in [(2, 1), (2, self.u.shape[-2])]:
            raise ValueError(f"_origin has invalid shape")

    @property
    def origin(self) -> Array | np.ndarray:
        """
        The shift of the sampling place, such that it is no longer centered at
        the origin. Defined as an array of shape ``(2 1... 1 1 C 1)``
        specifying the shift in the y and x directions respectively.
        """
        return _broadcast_2d_to_grid(self._origin, self.ndim)

    @property
    def extent(self) -> Array | np.ndarray:
        """
        The extent (lengths in height and width per wavelength) of the field
        in units of distance. Defined as an array of shape ``(2 1... 1 1 C 1)``
        specifying the extent in the y and x dimensions respectively.
        """
        shape = jnp.array(self.spatial_shape)
        shape = _broadcast_1d_to_grid(shape, self.ndim)
        return self.dx * shape

    @property
    def spectrum(self) -> Array | np.ndarray:
        """
        Wavelengths sampled by the complex field, shape ``(1... 1 1 C 1)``.
        """
        return _broadcast_1d_to_channels(self._spectrum, self.ndim)

    @property
    def spectral_density(self) -> Array | np.ndarray:
        """
        Weights of wavelengths sampled by the complex field, shape ``(1... 1 1
        C 1)``.
        """
        return _broadcast_1d_to_channels(self._spectral_density, self.ndim)

    @property
    def phase(self) -> Array:
        """
        Phase of the complex field, shape `(B... H W C [1 | 3])`.
        """
        return jnp.angle(self.u)

    @property
    def amplitude(self) -> Array:
        """
        Amplitude of the complex field, shape `(B... H W C [1 | 3])`. This is
        actually what is called the "magnitude".
        """
        return jnp.abs(self.u)

    @property
    def intensity(self) -> Array:
        """Intensity of the complex field, shape `(B... H W 1 1)`."""
        return jnp.sum(
            self.spectral_density * jnp.abs(self.u) ** 2, axis=(-2, -1), keepdims=True
        )

    @property
    def power(self) -> Array:
        """Power of the complex field, shape `(B... 1 1 1)`."""
        area = jnp.prod(self.dx, axis=0, keepdims=False)
        return jnp.sum(self.intensity, axis=(-4, -3), keepdims=True) * area

    @property
    def spatial_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Return the spatial limits of the field: (y_min, y_max), (x_min, x_max).
        """
        return (float(self.grid[0].min()), float(self.grid[0].max())), (
            float(self.grid[1].min()),
            float(self.grid[1].max()),
        )


class ScalarField(Field):

    @classmethod
    def create(
        cls,
        dx: ArrayLike,
        spectrum: ArrayLike,
        spectral_density: ArrayLike,
        u: ArrayLike | None = None,
        shape: tuple[int, int] | None = None,
        origin: ArrayLike | None = None,
    ) -> Self:
        """
        Create a ``ScalarField`` instance in a convenient way.

        Parameters
        ----------
        dx : array_like
            Sample spacing. Must be a scalar or an array with shape ``(2,)``,
            ``(2, 1)``, or ``(2, 1, 1)``. See the ``Raster`` class for further details.
        spectrum : array_like
            Wavelengths sampled by the field. Must be a scalar or a one-dimensional
            array of length ``C``.
        spectral_density : array_like
            Weights associated with the sampled wavelengths. Must be a scalar
            or a one-dimensional array of length ``C``.
        u : array_like, optional
            Scalar field values with shape ``(B..., H, W, C, 1)``. If not provided,
            the field is initialized with zeros using the specified ``shape``.
        shape : tuple of int, optional
            Spatial dimensions ``(H, W)`` of the field. Ignored if ``u`` is
            provided. Required if ``u`` is not provided.
        origin : array_like, optional
            Offset of the sampling plane relative to the origin. Must have
            shape ``(2,)``, ``(2, 1)`` or ``(2, C)``.

        Returns
        -------
        ScalarField
            A new instance with appropriately shaped internal arrays.
        """
        return cls._create(1, dx, spectrum, spectral_density, u, shape, origin)

    @override
    def _check_shapes(self):
        super()._check_shapes()
        if self.u.shape[-1] != 1:
            raise ValueError("last axis of u must have size 1 for ScalarField")


class VectorField(Field):

    @classmethod
    def create(
        cls,
        dx: ArrayLike,
        spectrum: ArrayLike,
        spectral_density: ArrayLike,
        u: ArrayLike | None = None,
        shape: tuple[int, int] | None = None,
        origin: ArrayLike | None = None,
    ) -> Self:
        """
        Create a ``VectorField`` instance in a convenient way.

        Parameters
        ----------
        dx : array_like
            Sample spacing. Must be a scalar or an array with shape ``(2,)``,
            ``(2, 1)``, or ``(2, 1, 1)``. See the ``Raster`` class for further details.
        spectrum : array_like
            Wavelengths sampled by the field. Must be a scalar or a one-dimensional
            array of length ``C``.
        spectral_density : array_like
            Weights associated with the sampled wavelengths. Must be a scalar
            or a one-dimensional array of length ``C``.
        u : array_like, optional
            Vector field of shape ``(B..., H, W, C, 3)``. If not provided,
            the field is initialized with zeros using the specified ``shape``.
        shape : tuple of int, optional
            Spatial dimensions ``(H, W)`` of the field. Ignored if ``u`` is
            provided. Required if ``u`` is not provided.
        origin : array_like, optional
            Offset of the sampling plane relative to the origin. Must have
            shape ``(2,)``, ``(2, 1)`` or ``(2, C)``.

        Returns
        -------
        VectorField
            A new instance with appropriately shaped internal arrays.
        """
        return cls._create(3, dx, spectrum, spectral_density, u, shape, origin)


    @property
    def jones_vector(self) -> Array:
        """Return Jones vector of field."""
        norm = jnp.linalg.norm(self.u, axis=-1, keepdims=True)
        norm = jnp.where(norm == 0, 1, norm)  # set to 1 to avoid division by zero
        return self.u / norm

    @override
    def _check_shapes(self):
        super()._check_shapes()
        if self.u.shape[-1] != 3:
            raise ValueError("last axis of u must have size 3 for VectorField")


def pad(field: Field, pad_width: int | tuple[int, int], cval: float = 0) -> Field:
    """
    Pad the `field` with zeros in one or two dimensions.
    Args:
        field: The field to pad.
        pad_width: The number of pixels to pad the field with.
        cval: The value to pad the field with (defauls is zero).
    """
    if isinstance(pad_width, int):
        pad_width = (pad_width, pad_width)
    u = jnp.pad(
        field.u,
        [(n, n) for n in (0,) * (field.ndim - 4) + (*pad_width, 0, 0)],
        constant_values=cval,
    )
    return field.replace(u=u)


def crop(field: Field, crop_width: int | tuple[int, int]) -> Field:
    """
    Crop the `field` by removing pixels from the edges.
    Args:
        field: The field to crop.
        crop_width: The number of pixels to remove from the edges.
    """
    if isinstance(crop_width, int):
        crop_width = (crop_width, crop_width)
    crop = [
        slice(n, size - n)
        for size, n in zip(field.shape, (0,) * (field.ndim - 4) + (*crop_width, 0, 0))
    ]
    return field.replace(u=field.u[tuple(crop)])


def shift_grid(field: Field, shift_yx: ArrayLike) -> Field:
    """
    Shift the sampling grid by an arbitrary amount in y and x directions.
    Args:
        shift_yx: The shift in y and x directions. Should be an array of
            shape `[2,]` in the format `[y, x]`.
    """
    if isinstance(shift_yx, Number):
        shift_yx = jnp.array([shift_yx, shift_yx])
    shift_yx = jnp.array(shift_yx)  # Ensure it is an array
    if shift_yx.ndim == 1:
        shift_yx = shift_yx[:, None]
    assert_rank(shift_yx, 2)
    return field.replace(_origin=shift_yx)


def shift_field(field: Field, shiftby: int | tuple[int, int]) -> Field:
    """
    Shift `field` by an integer number of pixels in one or two dimensions,
    while keeping the sampling grid centered at the origin.

    Args:
        field: The field to shift.
        shiftby: The number of pixels to shift the field by.

    See also shift_ft for subpixel shifts.
    """
    if isinstance(shiftby, int):
        shiftby = (shiftby, shiftby)

    crop = [
        (slice(n, dsize) if (n > 0) else slice(0, dsize + n))
        for dsize, n in zip(field.shape, (0,) * (field.ndim - 4) + (*shiftby, 0, 0))
    ]

    pads = [
        ((0, n) if (n > 0) else (-n, 0))
        for n in ((0,) * (field.ndim - 4) + (*shiftby, 0, 0))
    ]
    u = jnp.pad(field.u[tuple(crop)], pads)

    return field.replace(u=u)


def cartesian_to_spherical(field: Field, n: float, NA: float, f: float) -> Array:
    """
    Converts the field to a spherical basis. This is useful for high NA lenses.

    Args:
        field: The incoming ``Field`` in pupil space, in Cartesian coordinates.
        n: Refractive index of the lens.
        NA: NA of the lens.
        f: Focal length of the lens.

    Returns:
        The Field.u in spherical coordinates.
        !!! warning
            Caution: does NOT return a full Field object.
    """
    pupil_radius = f * NA / n
    mask = field.grid[0] ** 2 + field.grid[1] ** 2 <= pupil_radius**2
    sin_theta2 = jnp.sum(field.grid**2, axis=0) * mask / f**2
    cos_theta = jnp.sqrt(1 - sin_theta2)
    sin_theta = jnp.sqrt(sin_theta2)

    phi = jnp.arctan2(field.grid[0], field.grid[1])
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    sin_2phi = 2 * sin_phi * cos_phi
    cos_2phi = cos_phi**2 - sin_phi**2

    field_x = field.u[:, :, :, :, 2][..., None]
    field_y = field.u[:, :, :, :, 1][..., None]

    # Source: Eq. (6) of arXiv:2502.03170
    e_inf_x = ((cos_theta + 1.0) + (cos_theta - 1.0) * cos_2phi) * field_x + (
        cos_theta - 1.0
    ) * sin_2phi * field_y
    e_inf_y = ((cos_theta + 1.0) - (cos_theta - 1.0) * cos_2phi) * field_y + (
        cos_theta - 1.0
    ) * sin_2phi * field_x
    e_inf_z = -2.0 * sin_theta * (cos_phi * field_x + sin_phi * field_y)

    return jnp.stack([e_inf_z, e_inf_y, e_inf_x], axis=-1).squeeze(-2) / 2
