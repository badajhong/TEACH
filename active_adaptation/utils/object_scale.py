"""Keep an object's lower surface at the same height when its size changes."""

import torch


def support_height_offset(
    points: torch.Tensor, quaternion_wxyz: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Return the world-Z root shift that preserves the unscaled lowest point.

    ``points`` is [N, P, 3] in the object's local frame. This also handles
    anisotropic scale and a tilted object; only a translation in world Z is
    returned, so its X/Y reference position stays unchanged.
    """
    w, x, y, z = quaternion_wxyz.unbind(-1)
    row_z = torch.stack((
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x.square() + y.square()),
    ), dim=-1)
    unscaled_min = (points * row_z[:, None, :]).sum(-1).amin(-1)
    scaled_min = (points * scale[:, None, :] * row_z[:, None, :]).sum(-1).amin(-1)
    return unscaled_min - scaled_min
