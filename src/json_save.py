import json

def tree_to_dict(node):
    """
    递归将八叉树节点转换为字典格式
    """
    if node is None:
        return None
    
    # 转换当前节点数据
    node_dict = {
        "depth": node.depth,
        "size": float(node.size),
        "center": node.center.tolist(),
        "gaussian": {
            "pos": node.gaussian["pos"].tolist(),
            "rot": [float(x) for x in node.gaussian["rot"]],
            "scale": [float(x) for x in node.gaussian["scale"]],
            "f_dc": [float(x) for x in node.gaussian["f_dc"]],
            "opacity": 1.0
        },
        "children": []
    }
    
    # 递归转换子节点
    for child in node.children:
        if child is not None:
            node_dict["children"].append(tree_to_dict(child))
        else:
            node_dict["children"].append(None)
            
    return node_dict

def export_tree_to_json(root, filename):
    """
    导出整棵树结构
    """
    print(f"正在转换树结构并导出 JSON...")
    tree_data = tree_to_dict(root)
    with open(filename, 'w') as f:
        json.dump(tree_data, f, indent=2)
    print(f"八叉树结构已保存至: {filename}")