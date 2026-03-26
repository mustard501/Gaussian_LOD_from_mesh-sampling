import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from gs_save import save_gaussian_ply
from json_save import export_tree_to_json

DMAX = 15
OUTPUT_DIR = "assets/outputs/"

class GaussianNode:
    def __init__(self, center, size, depth):
        self.center = center    # 节点的几何中心 (AABB center)
        self.size = size        # 节点的边长
        self.depth = depth      # 当前深度
        self.points_indices = [] # 落在该节点内的点索引
        self.children = [None] * 8
        self.gaussian = None    # 存储计算出的高斯参数

def compute_gaussian_params(points, colors, node_size):
    """
    根据节点内的点云计算 3DGS 参数
    """

    # 1. 位置：重心
    pos = np.mean(points, axis=0)

    # 2. 方向：PCA
    # 计算协方差矩阵
    centered_points = points - pos
    cov = np.dot(centered_points.T, centered_points) / (len(points) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    idx = np.argsort(eigenvalues)[::-1]
    e = eigenvalues[idx]
    v = eigenvectors[:, idx]

    # 构造旋转矩阵并转换为四元数
    # 确保v是右手系
    if np.linalg.det(v) < 0:
        v[:, 2] *= -1

    r = R.from_matrix(v)
    q = r.as_quat()
    rot_q = [q[3], q[0], q[1], q[2]]

    # # eigenvectors[:, 0] 是特征值最小的轴，即法线方向
    # # 我们需要将原始高斯 (0,0,1) 轴旋转到这个法线方向
    normal = eigenvectors[:, 0]

    # 3. 尺寸
    stds = np.sqrt(np.maximum(e, 1e-10))

    minp = points.min(axis=0)
    maxp = points.max(axis=0)
    sx = max(np.max(maxp - minp),1e-8) * 0.7
    sy = sx * max(stds[1]/stds[0], 0.3)
    sz = sx * np.clip(stds[2]/stds[0], 0.05,0.1)
    scale = np.log(np.array([sx, sy, sz]))

    # 4. 颜色
    avg_rgb = np.mean(colors, axis=0)
    f_dc = (avg_rgb - 0.5) / 0.28209

    return {"norm": normal, "pos": pos, "rot": rot_q, "scale": scale, 'f_dc':f_dc}

def build_octree(points, colors, center, size, depth, max_depth, min_points):
    """
    递归构建八叉树
    """
    node = GaussianNode(center, size, depth)
    
    # 计算当前节点的高斯
    node.gaussian = compute_gaussian_params(points, colors, size)
    
    if depth >= max_depth or len(points) <= min_points:
        return node

    # 划分 8 个子节点
    half = size / 4
    offsets = [
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ]
    
    for i, offset in enumerate(offsets):
        child_center = center + np.array(offset) * half
        # 筛选落在子节点范围内的点
        mask = np.all(np.abs(points - child_center) <= size/4, axis=1)
        child_points = points[mask]
        child_colors = colors[mask]
        
        if len(child_points) >= 10:
            node.children[i] = build_octree(child_points, child_colors,  child_center, size/2, depth+1, max_depth, min_points)
            
    return node

def extract_lod(node, target_depth, result_list):
    """
    提取特定深度的所有高斯
    """
    if node is None: return
    if node.depth == target_depth:
        result_list.append(node.gaussian)
        return
    
    for child in node.children:
        if child:
            extract_lod(child, target_depth, result_list)
        elif abs(node.depth - target_depth) <= 1:
            result_list.append(node.gaussian)

if __name__ == "__main__":
    # 1. 假设已经有了点云 pcd (来自第一步)
    pcd = o3d.io.read_point_cloud("assets/pcds/textured_tree.ply")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # 2. 初始化八叉树参数
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2
    size = np.max(max_bound - min_bound)

    # 3. 构建树
    print("正在构建八叉树并计算高斯层次...")
    root = build_octree(points, colors, center, size, depth=0, max_depth=DMAX, min_points=5)

    # 4. 导出不同 LOD 的模型
    for d in [5,6,7,8,9]:
        lod_gaussians = []
        extract_lod(root, d, lod_gaussians)
        if d > DMAX:
            print(f"The {d}th model has not been created.")
        elif lod_gaussians:
            save_gaussian_ply(lod_gaussians, f"{OUTPUT_DIR}/lod_depth_{d}.ply")
    # # 5. 导出完整的树结构 JSON
    export_tree_to_json(root, f"{OUTPUT_DIR}/gaussian_octree_structure.json")