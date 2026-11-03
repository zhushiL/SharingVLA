import torch
from torch import Tensor

__all__ = [
    '_is_pose_relative_type',
    'relative_pose',
    'absolute_pose',
]

def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternion in xyzw format. Shape: (..., 4)."""
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(eps)
    return q / norm


def quat_canonicalize(q: torch.Tensor) -> torch.Tensor:
    """Flip quaternion sign so the scalar part keeps a stable non-negative sign."""
    sign = torch.where(q[..., 3:4] < 0, -1.0, 1.0)
    return q * sign


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """Inverse of quaternion in xyzw format. Shape: (..., 4)."""
    inv = quat_normalize(q).clone()
    inv[..., :3] = -inv[..., :3]  # negate xyz (imaginary part)
    return inv


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 * q2 in xyzw format. Shape: (..., 4)"""
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate 3D vector(s) by quaternion(s) in xyzw format."""
    q = quat_normalize(q)
    v_quat = torch.cat([v, torch.zeros_like(v[..., :1])], dim=-1)
    return quat_multiply(quat_multiply(q, v_quat), quat_inverse(q))[..., :3]


def quat_rotate_inverse(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate 3D vector(s) by the inverse of quaternion(s) in xyzw format."""
    return quat_rotate(quat_inverse(q), v)


def _resolve_pose_relative_type(relative_type: str | None) -> str:
    if relative_type in (None, 'world'):
        return 'world'
    if relative_type in ('local',):
        return 'local'
    raise ValueError(f"Unsupported pose relative type: {relative_type}")


def _is_pose_relative_type(relative_type: str | None) -> bool:
    return relative_type in {'world', 'local'}


def rotation_6d_to_matrix(rotation_6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert two concatenated rotation-matrix columns to a rotation matrix.

    ``rotation_6d`` is laid out as ``[R[:, 0], R[:, 1]]``. Gram-Schmidt
    orthogonalization keeps model predictions on SO(3), even when their six
    values are not perfectly orthonormal.
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(
            f"Expected a 6D rotation with shape (..., 6), got {tuple(rotation_6d.shape)}."
        )

    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:]
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    fallback_first = torch.zeros_like(first)
    fallback_first[..., 0] = 1
    first = torch.where(
        first_norm > eps,
        first / first_norm.clamp_min(eps),
        fallback_first,
    )

    second = second - (first * second).sum(dim=-1, keepdim=True) * first
    second_norm = torch.linalg.vector_norm(second, dim=-1, keepdim=True)
    # For a zero or collinear second column, select the canonical axis least
    # aligned with the first column and project it into the orthogonal plane.
    fallback_axis = torch.eye(3, dtype=first.dtype, device=first.device)[
        first.abs().argmin(dim=-1)
    ]
    fallback_second = fallback_axis - (
        first * fallback_axis
    ).sum(dim=-1, keepdim=True) * first
    fallback_second = fallback_second / torch.linalg.vector_norm(
        fallback_second, dim=-1, keepdim=True
    ).clamp_min(eps)
    second = torch.where(
        second_norm > eps,
        second / second_norm.clamp_min(eps),
        fallback_second,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Return a rotation matrix's first two columns in concatenated form."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected a rotation matrix with shape (..., 3, 3), got {tuple(matrix.shape)}."
        )
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def _resolve_pose_dim(total_dim: int, pose_dim: int | None) -> int:
    """Resolve xyz+quaternion (7D) versus xyz+rotation-6D (9D)."""
    if pose_dim is not None:
        if pose_dim not in (7, 9):
            raise ValueError(f"pose_dim must be 7 or 9, got {pose_dim}.")
        if total_dim % pose_dim != 0:
            raise ValueError(
                f"Pose feature dimension {total_dim} is not divisible by pose_dim={pose_dim}."
            )
        return pose_dim

    # Prefer 7D when both divide the feature width, preserving the historical
    # quaternion behavior for the ambiguous (and very unusual) 63D case.
    if total_dim % 7 == 0:
        return 7
    if total_dim % 9 == 0:
        return 9
    raise ValueError(
        f"Cannot infer pose representation from feature dimension {total_dim}; "
        "expected concatenated 7D xyz+quaternion or 9D xyz+rotation-6D poses."
    )


def _validate_pose_pair(first: torch.Tensor, second: torch.Tensor) -> int:
    if first.shape[-1] != second.shape[-1]:
        raise ValueError(
            "Action and state pose dimensions must match, got "
            f"{first.shape[-1]} and {second.shape[-1]}."
        )
    return first.shape[-1]


def relative_pose(
    action: torch.Tensor,
    state: torch.Tensor,
    pose_dim: int | None = None,
    relative_type: str | None = 'world',
) -> torch.Tensor:
    """Compute relative poses for quaternion or rotation-6D representations.

    A pose is either ``xyz + quaternion(xyzw)`` (7D) or ``xyz + the first two
    rotation-matrix columns`` (9D). Concatenated multi-arm poses are supported.
    """
    relative_type = _resolve_pose_relative_type(relative_type)
    total_dim = _validate_pose_pair(action, state)
    pose_dim = _resolve_pose_dim(total_dim, pose_dim)
    chunks = total_dim // pose_dim
    parts = []
    for i in range(chunks):
        s = i * pose_dim
        a_xyz = action[..., s:s+3]
        s_xyz = state[..., s:s+3]
        if pose_dim == 7:
            a_q = quat_normalize(action[..., s+3:s+pose_dim])
            s_q = quat_normalize(state[..., s+3:s+pose_dim])
            if relative_type == 'local':
                rel_xyz = quat_rotate_inverse(a_xyz - s_xyz, s_q)
            else:
                rel_xyz = a_xyz - s_xyz
            rel_rotation = quat_canonicalize(
                quat_normalize(quat_multiply(quat_inverse(s_q), a_q))
            )
        else:
            a_matrix = rotation_6d_to_matrix(action[..., s+3:s+pose_dim])
            s_matrix = rotation_6d_to_matrix(state[..., s+3:s+pose_dim])
            if relative_type == 'local':
                rel_xyz = torch.matmul(
                    s_matrix.transpose(-1, -2), (a_xyz - s_xyz).unsqueeze(-1)
                ).squeeze(-1)
            else:
                rel_xyz = a_xyz - s_xyz
            rel_matrix = torch.matmul(s_matrix.transpose(-1, -2), a_matrix)
            rel_rotation = matrix_to_rotation_6d(rel_matrix)
        parts.append(torch.cat([rel_xyz, rel_rotation], dim=-1))
    return torch.cat(parts, dim=-1)


def absolute_pose(
    rel_action: torch.Tensor,
    state: torch.Tensor,
    pose_dim: int | None = None,
    relative_type: str | None = 'world',
) -> torch.Tensor:
    """Recover absolute quaternion or rotation-6D poses from relative poses.

    This is the inverse of :func:`relative_pose`, apart from the
    intentional orthogonalization of rotation-6D inputs and predictions.
    """
    relative_type = _resolve_pose_relative_type(relative_type)
    total_dim = _validate_pose_pair(rel_action, state)
    pose_dim = _resolve_pose_dim(total_dim, pose_dim)
    chunks = total_dim // pose_dim
    parts = []
    for i in range(chunks):
        s = i * pose_dim
        r_xyz = rel_action[..., s:s+3]
        s_xyz = state[..., s:s+3]
        if pose_dim == 7:
            r_q = quat_normalize(rel_action[..., s+3:s+pose_dim])
            s_q = quat_normalize(state[..., s+3:s+pose_dim])
            if relative_type == 'local':
                abs_xyz = s_xyz + quat_rotate(s_q, r_xyz)
            else:
                abs_xyz = s_xyz + r_xyz
            abs_rotation = quat_canonicalize(
                quat_normalize(quat_multiply(s_q, r_q))
            )
        else:
            r_matrix = rotation_6d_to_matrix(rel_action[..., s+3:s+pose_dim])
            s_matrix = rotation_6d_to_matrix(state[..., s+3:s+pose_dim])
            if relative_type == 'local':
                abs_xyz = s_xyz + torch.matmul(s_matrix, r_xyz.unsqueeze(-1)).squeeze(-1)
            else:
                abs_xyz = s_xyz + r_xyz
            abs_matrix = torch.matmul(s_matrix, r_matrix)
            abs_rotation = matrix_to_rotation_6d(abs_matrix)
        parts.append(torch.cat([abs_xyz, abs_rotation], dim=-1))
    return torch.cat(parts, dim=-1)


def matrix_to_quat(R: Tensor) -> Tensor:
    """Convert rotation matrix (..., 3, 3) to quaternion (..., 4) in xyzw format.

    Uses Shepperd's method for numerical stability.
    """
    batch_shape = R.shape[:-2]
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    trace = m00 + m11 + m22
    q = torch.zeros(*batch_shape, 4, dtype=R.dtype, device=R.device)

    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2
    q[..., 3] = 0.25 * s
    q[..., 0] = (m21 - m12) / s
    q[..., 1] = (m02 - m20) / s
    q[..., 2] = (m10 - m01) / s

    cond2 = (m00 > m11) & (m00 > m22) & (trace <= 0)
    s2 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-10)) * 2
    q[..., 0] = torch.where(cond2, 0.25 * s2, q[..., 0])
    q[..., 1] = torch.where(cond2, (m01 + m10) / s2, q[..., 1])
    q[..., 2] = torch.where(cond2, (m02 + m20) / s2, q[..., 2])
    q[..., 3] = torch.where(cond2, (m21 - m12) / s2, q[..., 3])

    cond3 = (~cond2) & (m11 > m22) & (trace <= 0)
    s3 = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-10)) * 2
    q[..., 0] = torch.where(cond3, (m01 + m10) / s3, q[..., 0])
    q[..., 1] = torch.where(cond3, 0.25 * s3, q[..., 1])
    q[..., 2] = torch.where(cond3, (m12 + m21) / s3, q[..., 2])
    q[..., 3] = torch.where(cond3, (m02 - m20) / s3, q[..., 3])

    cond4 = (~cond2) & (~cond3) & (trace <= 0)
    s4 = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-10)) * 2
    q[..., 0] = torch.where(cond4, (m02 + m20) / s4, q[..., 0])
    q[..., 1] = torch.where(cond4, (m12 + m21) / s4, q[..., 1])
    q[..., 2] = torch.where(cond4, 0.25 * s4, q[..., 2])
    q[..., 3] = torch.where(cond4, (m10 - m01) / s4, q[..., 3])

    return quat_normalize(q)
