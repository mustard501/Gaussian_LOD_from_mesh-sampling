import argparse
import os

import numpy as np
import trimesh


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


def _get_base_color_texture(material):
    if material is None:
        return None
    if hasattr(material, "baseColorTexture") and material.baseColorTexture is not None:
        return material.baseColorTexture
    if hasattr(material, "image") and material.image is not None:
        return material.image
    return None


def _sample_base_color_from_texture(image, uvs):
    """从 baseColor 贴图采样 RGB 三通道，忽略 alpha。"""
    rgb_img = image.convert("RGB")
    w, h = rgb_img.size

    u = np.clip(uvs[:, 0], 0, 1)
    v = np.clip(uvs[:, 1], 0, 1)
    px = (u * (w - 1)).astype(int)
    py = ((1 - v) * (h - 1)).astype(int)

    rgb_arr = np.array(rgb_img)
    return rgb_arr[py, px] / 255.0


def sample_textured_point_cloud(mesh_path, output_pc_path, samples_per_face=2):
    """
    从带 baseColor 贴图的 Mesh 采样彩色点云。
    支持 .obj（含外部贴图）与 .glb / .gltf（内嵌贴图），仅读取 RGB 三通道。
    """
    geometries = _load_geometries(mesh_path)
    if not geometries:
        print("错误：未找到有效三角网格。")
        return

    total_faces = sum(len(mesh.faces) for mesh in geometries.values())
    num_samples_total = total_faces * samples_per_face
    print(f"检测到总面片数: {total_faces}, 目标采样点数: {num_samples_total}")

    all_points = []
    all_colors = []

    for name, mesh in geometries.items():
        n_samples = int(num_samples_total * (len(mesh.faces) / total_faces))
        if n_samples == 0:
            continue

        points, face_index = mesh.sample(n_samples, return_index=True)
        colors = np.ones((len(points), 3)) * 0.8

        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            face_uvs = mesh.visual.uv[mesh.faces[face_index]]
            barycentric = trimesh.triangles.points_to_barycentric(
                mesh.triangles[face_index],
                points,
            )
            uvs = (face_uvs * barycentric[:, :, np.newaxis]).sum(axis=1)

            image = _get_base_color_texture(getattr(mesh.visual, "material", None))
            if image is not None:
                colors = _sample_base_color_from_texture(image, uvs)
            else:
                print(f"警告: {name} 未找到 baseColor 贴图，使用默认灰色。")

        all_points.append(points)
        all_colors.append(colors)

    if not all_points:
        print("错误：没有采样到任何有效点。")
        return

    final_points = np.vstack(all_points)
    final_colors = np.vstack(all_colors)

    os.makedirs(os.path.dirname(output_pc_path) or ".", exist_ok=True)
    pcd = trimesh.points.PointCloud(vertices=final_points, colors=final_colors)
    pcd.export(output_pc_path)

    print(f"采样完成！有效点数: {len(final_points)}，点云已保存至: {output_pc_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="从带 baseColor 贴图的 Mesh 采样彩色点云")
    parser.add_argument("-i", "--input", required=True, help="输入 mesh 路径 (.obj / .glb / .gltf)")
    parser.add_argument("-o", "--output", required=True, help="输出点云路径 (.ply)")
    parser.add_argument(
        "--samples-per-face",
        type=int,
        default=2,
        help="每个三角面片的采样点数 (默认: 2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample_textured_point_cloud(args.input, args.output, samples_per_face=args.samples_per_face)
