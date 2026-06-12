"""
Static deformation utilities extracted from FEM_Model_simplified.ipynb.

The notebook computes static vertical and horizontal deformation by:
1. Building equivalent nodal load vectors Fz and Fy from uniform distributed loads.
2. Reducing them to the free DOFs.
3. Solving K_FF u_F = F_F.
4. Expanding the solution back to the full DOF vector.
5. Extracting nodal uy and uz values.

DOF order per node is assumed to be:
[ux, uy, uz, rx, ry, rz]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StaticDeformationResult:
    """Container for static deformation results."""

    uy_nodes: np.ndarray
    uz_nodes: np.ndarray
    u_y_full: np.ndarray
    u_z_full: np.ndarray
    u_static_full: np.ndarray
    Fy: np.ndarray
    Fz: np.ndarray
    F_static: np.ndarray
    element_lengths: np.ndarray
    element_length_factor: float


def _as_array(a, *, dtype=float) -> np.ndarray:
    return np.asarray(a, dtype=dtype)


def element_lengths(node_coordinates, elements) -> np.ndarray:
    """
    Compute element lengths from nodal coordinates and element connectivity.

    Parameters
    ----------
    node_coordinates : array_like, shape (n_nodes, 3)
        Node coordinates, equivalent to `NodeC` in the notebook.
    elements : array_like, shape (n_elements, 2)
        Element connectivity, equivalent to `Ele_tunnel` in the notebook.

    Returns
    -------
    lengths : ndarray, shape (n_elements,)
        Euclidean length of each element.
    """
    node_coordinates = _as_array(node_coordinates, dtype=float)
    elements = np.asarray(elements, dtype=int)

    if elements.ndim != 2 or elements.shape[1] < 2:
        raise ValueError("elements must have shape (n_elements, 2) or wider.")

    n1 = elements[:, 0]
    n2 = elements[:, 1]
    return np.linalg.norm(node_coordinates[n2] - node_coordinates[n1], axis=1)


def compute_static_deformations(
    K_FF,
    DofsF,
    nDof: int,
    elements,
    node_coordinates,
    *,
    LDOF: int = 6,
    qy: float = 11e3,
    qz: float = 250e3,
    mooring_restoring_force: float = 2900e3,
    element_length_factor: Optional[float] = None,
) -> StaticDeformationResult:
    """
    Compute static horizontal and vertical deformations using the notebook logic.

    Parameters
    ----------
    K_FF : array_like, shape (n_free_dofs, n_free_dofs)
        Reduced stiffness matrix for free DOFs.
    DofsF : array_like, shape (n_free_dofs,)
        Indices of free DOFs in the full system.
    nDof : int
        Total number of DOFs in the full system.
    elements : array_like, shape (n_elements, 2)
        Element connectivity, equivalent to `Ele_tunnel` in the notebook.
    node_coordinates : array_like, shape (n_nodes, 3)
        Node coordinates, equivalent to `NodeC` in the notebook.
    LDOF : int, default 6
        Number of local DOFs per node.
    qy : float, default 11e3
        Uniform horizontal load in y direction [N/m].
    qz : float, default 250e3
        Uniform vertical load in z direction [N/m].
    mooring_restoring_force : float, default 2900e3
        Restoring force used in the notebook vertical-load term [N].
    element_length_factor : float or None, default None
        Factor used in the notebook expression:
            qz * Le / 2 - mooring_restoring_force / element_length_factor

        If None, it is computed as 25 / first_element_length, matching the
        notebook's intended scaling while avoiding reliance on a previous
        global value of `Le`.

    Returns
    -------
    StaticDeformationResult
        Includes nodal uy/uz arrays, full displacement vectors, and force vectors.

    Notes
    -----
    This preserves the notebook's static load construction:
        Fz[node_z] += qz * Le / 2 - mooring_restoring_force / element_length_factor
        Fy[node_y] += qy * Le / 2
    """
    K_FF = _as_array(K_FF, dtype=float)
    DofsF = np.asarray(DofsF, dtype=int)
    elements = np.asarray(elements, dtype=int)
    node_coordinates = _as_array(node_coordinates, dtype=float)

    if K_FF.shape[0] != K_FF.shape[1]:
        raise ValueError("K_FF must be square.")
    if K_FF.shape[0] != len(DofsF):
        raise ValueError("K_FF size must match len(DofsF).")
    if nDof % LDOF != 0:
        raise ValueError("nDof must be divisible by LDOF.")

    lengths = element_lengths(node_coordinates, elements)
    if np.any(lengths <= 0):
        raise ValueError("All element lengths must be positive.")

    if element_length_factor is None:
        element_length_factor = 25.0 / float(lengths[0])

    Fy = np.zeros(nDof, dtype=float)
    Fz = np.zeros(nDof, dtype=float)

    for (n1, n2), Le in zip(elements[:, :2], lengths):
        # Equivalent nodal forces for uniform vertical load plus mooring term.
        Fz[n1 * LDOF + 2] += (qz * Le / 2.0) - (
            mooring_restoring_force / element_length_factor
        )
        Fz[n2 * LDOF + 2] += (qz * Le / 2.0) - (
            mooring_restoring_force / element_length_factor
        )

        # Equivalent nodal forces for uniform horizontal y load.
        Fy[n1 * LDOF + 1] += qy * Le / 2.0
        Fy[n2 * LDOF + 1] += qy * Le / 2.0

    u_z_free = np.linalg.solve(K_FF, Fz[DofsF])
    u_y_free = np.linalg.solve(K_FF, Fy[DofsF])

    u_z_full = np.zeros(nDof, dtype=float)
    u_y_full = np.zeros(nDof, dtype=float)
    u_z_full[DofsF] = u_z_free
    u_y_full[DofsF] = u_y_free

    F_static = Fy + Fz
    u_static_full = u_y_full + u_z_full

    uy_nodes = u_y_full[1::LDOF]
    uz_nodes = u_z_full[2::LDOF]

    return StaticDeformationResult(
        uy_nodes=uy_nodes,
        uz_nodes=uz_nodes,
        u_y_full=u_y_full,
        u_z_full=u_z_full,
        u_static_full=u_static_full,
        Fy=Fy,
        Fz=Fz,
        F_static=F_static,
        element_lengths=lengths,
        element_length_factor=float(element_length_factor),
    )


def plot_static_deformations(s, result: StaticDeformationResult):
    """
    Plot static uy and uz deformation curves.

    Parameters
    ----------
    s : array_like
        Beam coordinate values, equivalent to `s` in the notebook.
    result : StaticDeformationResult
        Output from `compute_static_deformations`.
    """
    import matplotlib.pyplot as plt

    s = np.asarray(s, dtype=float)

    plt.figure()
    plt.plot(s, result.uz_nodes, "-o")
    plt.xlabel("s [m]")
    plt.ylabel("Vertical displacement uz [m]")
    plt.title("Static vertical displacement")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(s, result.uy_nodes, "-o")
    plt.xlabel("s [m]")
    plt.ylabel("Horizontal displacement uy [m]")
    plt.title("Static horizontal displacement")
    plt.grid(True)
    plt.show()
