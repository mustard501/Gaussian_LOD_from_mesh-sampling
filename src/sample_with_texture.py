import trimesh
import numpy as np
from PIL import Image
import os

def sample_textured_point_cloud(obj_path, output_pc_path):
    # 1. 加载模型
    scene_or_mesh = trimesh.load(obj_path, process=False)
    
    # 统一处理 Scene 和单个 Mesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        geometries = scene_or_mesh.geometry
    else:
        geometries = {'main': scene_or_mesh}

    # 计算总面数用于比例分配
    total_faces = sum(len(m.faces) for m in geometries.values() if hasattr(m, 'faces'))
    num_samples_total = total_faces * 2
    print(f"检测到总面片数: {total_faces}, 目标采样点数: {num_samples_total}")

    all_points = []
    all_colors = []
    all_normals = []

    # 2. 遍历所有子几何体
    for name, mesh in geometries.items():
        if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            continue
        
        n_samples = int(num_samples_total * (len(mesh.faces) / total_faces))
        if n_samples == 0: continue

        # 采样点、对应的面索引
        points, face_index = mesh.sample(n_samples, return_index=True)
        normals = mesh.face_normals[face_index]
        
        # 默认颜色
        colors = np.ones((len(points), 3)) * 0.8 

        # --- 核心修改：手动插值 UV ---
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            # A. 获取采样点所在面片的三个顶点的 UV 坐标
            # mesh.faces[face_index] 拿到面片对应的顶点索引
            # mesh.visual.uv[...] 拿到这些顶点的 UV
            face_uvs = mesh.visual.uv[mesh.faces[face_index]] 

            # B. 计算采样点在三角形中的重心坐标 (Barycentric Coordinates)
            # trimesh 的 triangles 模块提供了这个工具
            barycentric = trimesh.triangles.points_to_barycentric(
                mesh.triangles[face_index], 
                points
            )

            # C. 插值得到采样点的 UV: (n, 3, 2) * (n, 3, 1) -> sum over axis 1
            uvs = (face_uvs * barycentric[:, :, np.newaxis]).sum(axis=1)

            # --- 获取材质贴图 ---
            material = mesh.visual.material
            # 兼容不同版本的 material 对象
            image = None
            if hasattr(material, 'image') and material.image is not None:
                image = material.image
            elif hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                image = material.baseColorTexture

            if image is not None:
                albedo_img = image.convert("RGB")
                w, h = albedo_img.size
                
                # UV 转像素坐标 (注意 V 轴通常需要翻转)
                u = np.clip(uvs[:, 0], 0, 1)
                v = np.clip(uvs[:, 1], 0, 1)
                px = (u * (w - 1)).astype(int)
                py = ((1 - v) * (h - 1)).astype(int)
                
                albedo_arr = np.array(albedo_img)
                colors = albedo_arr[py, px] / 255.0

        all_points.append(points)
        all_colors.append(colors)
        all_normals.append(normals)

    # 3. 合并
    if not all_points:
        print("错误：没有采样到任何有效点。")
        return

    final_points = np.vstack(all_points)
    final_colors = np.vstack(all_colors)
    final_normals = np.vstack(all_normals)

    # 4. 导出
    pcd = trimesh.points.PointCloud(vertices=final_points, colors=final_colors)
    # PointCloud 没有 vertices_normal 属性，但可以作为 metadata 存储或直接保存为 ply
    pcd.export(output_pc_path)
    
    print(f"采样完成！有效点数: {len(final_points)}，点云已保存至: {output_pc_path}")

if __name__ == "__main__":
    # 确保路径存在
    OBJ_FILE = "assets/inputs/tree/tree.obj"
    OUT_DIR = "assets/pcds"
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    OUT_PLY = os.path.join(OUT_DIR, "textured_tree.ply")
    sample_textured_point_cloud(OBJ_FILE, OUT_PLY)