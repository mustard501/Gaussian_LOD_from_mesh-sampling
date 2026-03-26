import struct
from plyfile import PlyData, PlyElement
import numpy as np

def get_sh_color(rgb_triplet):
    """
    将 [0, 1] 范围的 RGB 转换为 3DGS 的 f_dc 系数
    """
    return [(val - 0.5) / 0.28209 for val in rgb_triplet]

# 推荐的白模颜色方案
WHITE_MODEL_PALETTE = {
    "pure_white": [1.77, 1.77, 1.77],     # 纯白 (容易过曝)
    "classic_plaster": [1.41, 1.41, 1.41],# 经典石膏感 (RGB 0.9)
    "warm_white": [1.41, 1.38, 1.35],    # 暖白 (模拟纸模)
    "cool_gray": [1.06, 1.06, 1.15],      # 冷灰 (更有工业设计感)
    "black": [-1.772, -1.772, -1.772]      # 黑
}

# 使用示例
current_sh = WHITE_MODEL_PALETTE["black"]

def save_gaussian_ply(gaussians, filename):
    """
    将高斯参数列表保存为标准 3DGS PLY 格式
    """
    count = len(gaussians)

    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]

    elements = np.empty(count, dtype=dtype)

    pos = np.array([g['pos'] for g in gaussians], dtype='f4')
    norms = np.zeros((count, 3), dtype='f4')
    colors_dc = np.array([g['f_dc'] for g in gaussians], dtype='f4')
    opacities = np.full((count, 1), 10.0, dtype='f4')
    scales = np.array([g['scale'] for g in gaussians], dtype='f4')
    rots = np.array([g['rot'] for g in gaussians], dtype='f4')

    attributes = np.concatenate([
        pos, norms, colors_dc, opacities, scales, rots
    ], axis=1)

    elements[:] =list(map(tuple, attributes))
    
    # 写入ply文件
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(filename)

    print(f"LOD 模型已导出至: {filename}")