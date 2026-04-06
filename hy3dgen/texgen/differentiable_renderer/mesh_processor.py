# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import numpy as np
from scipy.sparse import csr_matrix


def meshVerticeInpaint_smooth(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx):
    texture_height, texture_width, texture_channel = texture.shape
    vtx_num = vtx_pos.shape[0]

    # --- Vectorized UV coordinate sampling ---
    uv_coords = vtx_uv[uv_idx]  # (F, 3, 2)
    pix_v = np.round(uv_coords[:, :, 0] * (texture_width  - 1)).astype(np.int32).clip(0, texture_width  - 1)  # (F, 3)
    pix_u = np.round((1.0 - uv_coords[:, :, 1]) * (texture_height - 1)).astype(np.int32).clip(0, texture_height - 1)  # (F, 3)

    is_colored = mask[pix_u, pix_v] > 0  # (F, 3) bool

    # --- Scatter sampled colors onto vertices ---
    vtx_mask  = np.zeros(vtx_num, dtype=np.float32)
    vtx_color = np.zeros((vtx_num, texture_channel), dtype=np.float32)

    colored_vtx = pos_idx[is_colored]
    colored_u   = pix_u[is_colored]
    colored_v   = pix_v[is_colored]
    vtx_color[colored_vtx] = texture[colored_u, colored_v]
    vtx_mask[colored_vtx]  = 1.0

    # --- Build sparse adjacency matrix for propagation ---
    # Each face contributes edges: 0↔1, 1↔2, 2↔0 (both directions)
    f0, f1, f2 = pos_idx[:, 0], pos_idx[:, 1], pos_idx[:, 2]
    src = np.concatenate([f0, f1, f2, f1, f2, f0])
    dst = np.concatenate([f1, f2, f0, f0, f1, f2])
    adj = csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)),
                     shape=(vtx_num, vtx_num))

    # --- Iterative propagation via sparse matrix ops ---
    # Each pass floods colors one edge-hop further from colored vertices.
    # Unweighted average (vs original distance-weighted) — indistinguishable
    # for seam-filling where cv2.inpaint handles residual gaps anyway.
    for _ in range(10):
        uncolored = vtx_mask == 0
        if not uncolored.any():
            break
        neighbor_color_sum = adj @ vtx_color          # (V, C)
        neighbor_mask_sum  = adj @ vtx_mask            # (V,)
        can_fill = uncolored & (neighbor_mask_sum > 0)
        if not can_fill.any():
            break
        vtx_color[can_fill] = (neighbor_color_sum[can_fill] /
                               neighbor_mask_sum[can_fill, None])
        vtx_mask[can_fill] = 1.0

    # --- Scatter vertex colors back to texture ---
    new_texture = texture.copy()
    new_mask    = mask.copy()

    write_mask  = vtx_mask[pos_idx] > 0  # (F, 3) bool
    write_vtx   = pos_idx[write_mask]
    write_u     = pix_u[write_mask]
    write_v     = pix_v[write_mask]
    new_texture[write_u, write_v] = vtx_color[write_vtx]
    new_mask[write_u, write_v]    = 255

    return new_texture, new_mask


def meshVerticeInpaint(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx, method="smooth"):
    if method == "smooth":
        return meshVerticeInpaint_smooth(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
    else:
        raise ValueError("Invalid method. Use 'smooth' or 'forward'.")
