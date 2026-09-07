import argparse
import os

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from gs_save import save_gaussian_ply
from json_save import export_tree_to_json


class GaussianNode:
    def __init__(self, center, size, depth):
        self.center = center
        self.size = size
        self.depth = depth
        self.points_indices = []
        self.children = [None] * 8
        self.gaussian = None


def compute_gaussian_params(points, colors, node_size):
    pos = np.mean(points, axis=0)

    centered_points = points - pos
    cov = np.dot(centered_points.T, centered_points) / (len(points) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    idx = np.argsort(eigenvalues)[::-1]
    e = eigenvalues[idx]
    v = eigenvectors[:, idx]

    if np.linalg.det(v) < 0:
        v[:, 2] *= -1

    r = R.from_matrix(v)
    q = r.as_quat()
    rot_q = [q[3], q[0], q[1], q[2]]
    normal = eigenvectors[:, 0]

    stds = np.sqrt(np.maximum(e, 1e-10))

    minp = points.min(axis=0)
    maxp = points.max(axis=0)
    sx = max(np.max(maxp - minp), 1e-8) * 0.7
    sy = sx * max(stds[1] / stds[0], 0.3)
    sz = sx * np.clip(stds[2] / stds[0], 0.05, 0.1)
    scale = np.log(np.array([sx, sy, sz]))

    avg_rgb = np.mean(colors, axis=0)
    f_dc = (avg_rgb - 0.5) / 0.28209

    return {"norm": normal, "pos": pos, "rot": rot_q, "scale": scale, "f_dc": f_dc}


def build_octree(points, colors, center, size, depth, max_depth, min_points):
    node = GaussianNode(center, size, depth)
    node.gaussian = compute_gaussian_params(points, colors, size)

    if depth >= max_depth or len(points) <= min_points:
        return node

    half = size / 4
    offsets = [
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ]

    for i, offset in enumerate(offsets):
        child_center = center + np.array(offset) * half
        mask = np.all(np.abs(points - child_center) <= size / 4, axis=1)
        child_points = points[mask]
        child_colors = colors[mask]

        if len(child_points) >= 10:
            node.children[i] = build_octree(
                child_points,
                child_colors,
                child_center,
                size / 2,
                depth + 1,
                max_depth,
                min_points,
            )

    return node


def extract_lod(node, target_depth, result_list):
    if node is None:
        return
    if node.depth == target_depth:
        result_list.append(node.gaussian)
        return

    for child in node.children:
        if child:
            extract_lod(child, target_depth, result_list)
        elif abs(node.depth - target_depth) <= 1:
            result_list.append(node.gaussian)


def run_octree_pipeline(input_pcd, output_dir, max_depth, min_points, lod_depths, json_path):
    os.makedirs(output_dir, exist_ok=True)

    pcd = o3d.io.read_point_cloud(input_pcd)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2
    size = np.max(max_bound - min_bound)

    print("正在构建八叉树并计算高斯层次...")
    root = build_octree(points, colors, center, size, depth=0, max_depth=max_depth, min_points=min_points)

    for d in lod_depths:
        lod_gaussians = []
        extract_lod(root, d, lod_gaussians)
        if d > max_depth:
            print(f"The {d}th model has not been created.")
        elif lod_gaussians:
            save_gaussian_ply(lod_gaussians, os.path.join(output_dir, f"lod_depth_{d}.ply"))

    export_tree_to_json(root, json_path)


def parse_args():
    parser = argparse.ArgumentParser(description="从点云构建八叉树并导出 3DGS LOD 模型")
    parser.add_argument("-i", "--input", required=True, help="输入点云路径 (.ply)")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--max-depth", type=int, default=15, help="八叉树最大深度 (默认: 15)")
    parser.add_argument(
        "--lod-depths",
        type=int,
        nargs="+",
        default=[5, 6, 7, 8, 9],
        help="要导出的 LOD 深度层级 (默认: 5 6 7 8 9)",
    )
    parser.add_argument("--min-points", type=int, default=5, help="节点停止细分的最少点数 (默认: 5)")
    parser.add_argument(
        "--json",
        default=None,
        help="八叉树 JSON 输出路径 (默认: <output>/gaussian_octree_structure.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    json_path = args.json or os.path.join(args.output, "gaussian_octree_structure.json")
    run_octree_pipeline(
        input_pcd=args.input,
        output_dir=args.output,
        max_depth=args.max_depth,
        min_points=args.min_points,
        lod_depths=args.lod_depths,
        json_path=json_path,
    )
