def load_config_from_file(file_path: str) -> dict:
    """从Python文件中加载配置字典"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 移除可能的注释和额外字符，只保留字典部分
        if content.startswith('{') and content.endswith('}'):
            # 处理1.py格式
            return eval(content)
        else:
            # 处理2.py格式
            # 提取字典内容
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                dict_str = content[start:end]
                return eval(dict_str)
    return {}


def compare_configs(config1: dict, config2: dict, path: str = "") -> list:
    """
    比较两个配置字典的差异

    Args:
        config1: 第一个配置字典
        config2: 第二个配置字典
        path: 当前路径（用于递归）

    Returns:
        差异列表，每个元素为(路径, config1中的值, config2中的值)
    """
    differences = []

    # 获取所有键
    all_keys = set(config1.keys()) | set(config2.keys())

    for key in all_keys:
        current_path = f"{path}.{key}" if path else key

        # 如果键只存在于一个配置中
        if key not in config1:
            differences.append((current_path, "KEY_MISSING", config2[key]))
            continue
        if key not in config2:
            differences.append((current_path, config1[key], "KEY_MISSING"))
            continue

        value1, value2 = config1[key], config2[key]

        # 如果值的类型不同
        if type(value1) != type(value2):
            differences.append((current_path, value1, value2))
            continue

        # 如果都是字典，递归比较
        if isinstance(value1, dict) and isinstance(value2, dict):
            differences.extend(compare_configs(value1, value2, current_path))
        # 如果都是列表，比较列表元素
        elif isinstance(value1, list) and isinstance(value2, list):
            if value1 != value2:
                differences.append((current_path, value1, value2))
        # 其他情况直接比较值
        elif value1 != value2:
            differences.append((current_path, value1, value2))

    return differences


def format_value(value) -> str:
    """格式化值用于显示"""
    if isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, (list, dict)):
        return str(value)
    else:
        return str(value)


def print_differences(differences: list) -> None:
    """打印差异"""
    if not differences:
        print("两个配置文件完全相同")
        return

    print(f"发现 {len(differences)} 处差异:\n")
    for path, value1, value2 in differences:
        print(f"路径: {path}")
        print(f"  配置1: {format_value(value1)}")
        print(f"  配置2: {format_value(value2)}")
        print("-" * 50)


def save_differences_to_file(differences: list, output_file: str) -> None:
    """将差异保存到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        if not differences:
            f.write("两个配置文件完全相同\n")
            return

        f.write(f"发现 {len(differences)} 处差异:\n\n")
        for path, value1, value2 in differences:
            f.write(f"路径: {path}\n")
            f.write(f"  配置1: {format_value(value1)}\n")
            f.write(f"  配置2: {format_value(value2)}\n")
            f.write("-" * 50 + "\n")


def main():
    # 加载两个配置文件
    try:
        config1 = load_config_from_file('1.py')
        config2 = load_config_from_file('3.py')
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        return

    # 比较配置
    differences = compare_configs(config1, config2)

    # 打印差异
    print_differences(differences)

    # 保存差异到文件
    save_differences_to_file(differences, 'config_differences.txt')
    print(f"\n差异已保存到 config_differences.txt")


if __name__ == "__main__":
    main()