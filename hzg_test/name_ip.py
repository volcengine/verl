import ray

# 初始化Ray
if not ray.is_initialized():
    ray.init()

# 获取所有节点的信息
nodes = ray.nodes()

# 打印表头
print(f"{'机器名':<20} {'IP地址':<15}")
print("-" * 40)

# 遍历所有节点并打印信息
for node in nodes:
    # 节点地址格式通常为 "IP:端口"，我们只需要IP部分
    ip_address = node["NodeManagerAddress"].split(":")[0]
    # 机器名（主机名）
    node_name = node["NodeManagerHostname"]

    print(f"{node_name:<20} {ip_address:<15}")