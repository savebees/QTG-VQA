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
validation_data = load_jsonl('results/sutd-traffic/validation_results.jsonl')

# 创建字典，根据 record_id 映射到 q_type 和答案
test_dict = {item[0]: {'q_type': item[5], 'answer': item[10]} for item in test_data}

# 计算每种问题类型的准确率
def calculate_accuracy(test_dict, validation_data):
    accuracy_dict = {}
    for item in validation_data:
        record_id = item['record_id']
        if record_id in test_dict:
            q_type = test_dict[record_id]['q_type']
            correct_answer = test_dict[record_id]['answer']
            predicted_answer = item['predicted_answer']

            if q_type not in accuracy_dict:
                accuracy_dict[q_type] = {'correct': 0, 'total': 0}

            accuracy_dict[q_type]['total'] += 1
            if predicted_answer == correct_answer:
                accuracy_dict[q_type]['correct'] += 1

    # 计算准确率
    for q_type, counts in accuracy_dict.items():
        total = counts['total']
        correct = counts['correct']
        accuracy_dict[q_type]['accuracy'] = (correct / total) if total > 0 else 0

    return accuracy_dict

# 执行准确率计算
accuracy_results = calculate_accuracy(test_dict, validation_data)
print(accuracy_results)