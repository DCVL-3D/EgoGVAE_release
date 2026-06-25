# 파일명: _base.py

import abc
import numpy as onp
import torch
from torch import Tensor

class MatrixLieGroup(abc.ABC):
    """Interface definition for matrix Lie groups."""

    # Class properties.
    matrix_dim: int
    parameters_dim: int
    tangent_dim: int
    space_dim: int

    def __init__(self, parameters: Tensor):
        """Construct a group object from its underlying parameters."""
        raise NotImplementedError()

    def __matmul__(self, other):
        """Overload for the `@` operator."""
        if isinstance(other, (onp.ndarray, Tensor)):
            return self.apply(target=other)
        elif isinstance(other, MatrixLieGroup):
            assert self.space_dim == other.space_dim
            return self.multiply(other=other)
        else:
            assert False, f"Invalid argument type for `@` operator: {type(other)}"

    # Factory.

    @classmethod
    @abc.abstractmethod
    def identity(cls, device: torch.device | str, dtype: torch.dtype):
        """Returns identity element."""

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls, matrix: Tensor):
        """Get group member from matrix representation."""

    # Accessors.

    @abc.abstractmethod
    def as_matrix(self) -> Tensor:
        """Get transformation as a matrix. Homogeneous for SE groups."""

    @abc.abstractmethod
    def parameters(self) -> Tensor:
        """Get underlying representation."""

    # Operations.

    @abc.abstractmethod
    def apply(self, target: Tensor) -> Tensor:
        """Applies group action to a point."""

    @abc.abstractmethod
    def multiply(self, other):
        """Composes this transformation with another."""

    @classmethod
    @abc.abstractmethod
    def exp(cls, tangent: Tensor):
        """Computes `expm(wedge(tangent))`."""

    @abc.abstractmethod
    def log(self) -> Tensor:
        """Computes `vee(logm(transformation matrix))`."""

    @abc.abstractmethod
    def adjoint(self) -> Tensor:
        """Computes the adjoint."""

    @abc.abstractmethod
    def inverse(self):
        """Computes the inverse of our transform."""

    @abc.abstractmethod
    def normalize(self):
        """Normalize/projects values and returns."""

    def get_batch_axes(self) -> tuple:
        """Return any leading batch axes in contained parameters."""
        return self.parameters().shape[:-1]


class SOBase(MatrixLieGroup):
    """Base class for special orthogonal groups."""


class SEBase(MatrixLieGroup):
    """Base class for special Euclidean groups."""

    # SE-specific interface.

    @classmethod
    @abc.abstractmethod
    def from_rotation_and_translation(cls, rotation, translation: Tensor):
        """Construct a rigid transform from a rotation and a translation."""

    @classmethod
    def from_rotation(cls, rotation):
        return cls.from_rotation_and_translation(
            rotation=rotation,
            translation=rotation.parameters().new_zeros(
                (*rotation.parameters().shape[:-1], cls.space_dim),
                dtype=rotation.parameters().dtype,
            ),
        )

    @abc.abstractmethod
    def rotation(self):
        """Returns a transform's rotation term."""

    @abc.abstractmethod
    def translation(self) -> Tensor:
        """Returns a transform's translation term."""

    # Overrides.

    def apply(self, target: Tensor) -> Tensor:
        return self.rotation() @ target + self.translation()

    def multiply(self, other):
        return type(self).from_rotation_and_translation(
            rotation=self.rotation() @ other.rotation(),
            translation=(self.rotation() @ other.translation()) + self.translation(),
        )

    def inverse(self):
        R_inv = self.rotation().inverse()
        return type(self).from_rotation_and_translation(
            rotation=R_inv,
            translation=-(R_inv @ self.translation()),
        )

    def normalize(self):
        return type(self).from_rotation_and_translation(
            rotation=self.rotation().normalize(),
            translation=self.translation(),
        )