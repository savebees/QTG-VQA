import json

# 从 JSONL 文件中读取数据
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 将JSON字符串转换为列表
            data.append(json.loads(line))
    return data

# 加载数据
test_data = load_jsonl('data/sutd-traffic/output_file_test.jsonl')

# 创建字典，用于统计各问题类型的频率
q_type_frequency = {}

for item in test_data:
    q_type = item[5]  # 根据索引位置获取 q_type
    if q_type in q_type_frequency:
        q_type_frequency[q_type] += 1
    else:
        q_type_frequency[q_type] = 1

# 打印各问题类型出现的频率
print(q_type_frequency)