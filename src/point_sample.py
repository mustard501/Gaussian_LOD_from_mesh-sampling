import argparse
import os

import numpy as np
import open3d as o3d
import trimesh

GLB_EXTENSIONS = {".glb", ".gltf"}


def _load_geometries(mesh_path):
    scene_or_mesh = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        return {
            name: mesh
            for name, mesh in scene_or_mesh.geometry.items()
            if hasattr(mesh, "faces") and len(mesh.faces) > 0
        }
    if hasattr(scene_or_mesh, "faces") and len(scene_or_mesh.faces) > 0:
        return {"main": scene_or_mesh}
    return {}


def _sample_geometries_to_cloud(geometries, samples_per_face):
    total_faces = sum(len(mesh.faces) for mesh in geometries.values())
    if total_faces == 0:
        return None

    num_samples_total = total_faces * samples_per_face
    print(f"检测到总面片数: {total_faces}, 目标采样点数: {num_samples_total}")

    all_points = []
    all_normals = []
    for mesh in geometries.values():
        n_samples = int(num_samples_total * (len(mesh.faces) / total_faces))
        if n_samples == 0:
            continue

        points, face_index = mesh.sample(n_samples, return_index=True)
        normals = mesh.face_normals[face_index]
        all_points.append(points)
        all_normals.append(normals)

    if not all_points:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    pcd.normals = o3d.utility.Vector3dVector(np.vstack(all_normals))
    return pcd


def sample_mesh_to_cloud(mesh_path, samples_per_face=2):
    """
    将 Mesh 采样为带法线的点云，支持 .obj / .ply / .stl / .glb / .gltf 等格式。
    采样密度 = 面片数 × samples_per_face（默认 2）。
    """
    if not os.path.exists(mesh_path):
        print(f"Error: 找不到文件 {mesh_path}")
        return None

    ext = os.path.splitext(mesh_path)[1].lower()

    if ext in GLB_EXTENSIONS:
        geometries = _load_geometries(mesh_path)
        if not geometries:
            print("Error: GLB/GLTF 中未找到有效三角网格。")
            return None
        return _sample_geometries_to_cloud(geometries, samples_per_face)

    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
    if len(mesh.triangles) == 0:
        print("Error: Mesh 不包含三角面片。")
        return None

    num_samples = len(mesh.triangles) * samples_per_face

    if not mesh.has_vertex_normals():
        print("正在计算顶点法线...")
        mesh.compute_vertex_normals()

    if not mesh.has_textures():
        print("警告: Mesh 没有检测到材质贴图，将使用顶点颜色或默认白模。")

    print(f"正在从 {len(mesh.triangles)} 个面片中采样 {num_samples} 个点...")
    return mesh.sample_points_uniformly(number_of_points=num_samples)


def save_point_cloud(pcd, output_path, show_preview=False):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存至: {output_path}")

    if show_preview:
        print("正在打开预览窗口...")
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Point Cloud Preview",
            width=1200,
            height=800,
            left=50,
            top=50,
            point_show_normal=False,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="将白模 Mesh 采样为带法线的点云")
    parser.add_argument("-i", "--input", required=True, help="输入 mesh 路径 (.obj / .glb / .gltf 等)")
    parser.add_argument("-o", "--output", required=True, help="输出点云路径 (.ply)")
    parser.add_argument(
        "--samples-per-face",
        type=int,
        default=2,
        help="每个三角面片的采样点数 (默认: 2)",
    )
    parser.add_argument("--preview", action="store_true", help="采样完成后打开预览窗口")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    point_cloud = sample_mesh_to_cloud(args.input, samples_per_face=args.samples_per_face)
    if point_cloud:
        save_point_cloud(point_cloud, args.output, show_preview=args.preview)
