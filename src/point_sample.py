import open3d as o3d
import numpy as np
import os

def sample_mesh_to_cloud(mesh_path, num_samples = 2000000):
    """
    将 Mesh 采样为带法线的点云
    :param mesh_path: 模型路径 (.obj, .stl, .ply 等)
    :param num_samples: 采样点数量，建议略大于面片数
    """
    # 1. 加载 Mesh
    if not os.path.exists(mesh_path):
        print(f"Error: 找不到文件 {mesh_path}")
        return None
    
    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
    num_samples = 2*len(mesh.triangles)
    
    # 检查是否有顶点法线，没有则计算
    if not mesh.has_vertex_normals():
        print("正在计算顶点法线...")
        mesh.compute_vertex_normals()

    if not mesh.has_textures():
        print("警告: Mesh 没有检测到材质贴图，将使用顶点颜色或默认白模。")

    # 2. 面积加权采样 (Uniform Sampling)
    print(f"正在从 {len(mesh.triangles)} 个面片中采样 {num_samples} 个点...")
    pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
    
    # 注意：sample_points_uniformly 会自动通过插值计算出每个采样点的法线 (Normals)
    
    return pcd

def save_and_visualize(pcd, output_path="assets/pcds/tree.ply"):
    # 保存为 PLY 格式，包含位置和法线
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存至: {output_path}")

    # 简单可视化
    print("正在打开预览窗口...")
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Preview",
                                      width=1200, height=800,
                                      left=50, top=50,
                                      point_show_normal=False)


if __name__ == "__main__":
    # 替换为你自己的 mesh 路径
    INPUT_MESH = "assets/inputs/tree.obj" 
    
    # 执行采样
    point_cloud = sample_mesh_to_cloud(INPUT_MESH)
    
    if point_cloud:
        save_and_visualize(point_cloud)