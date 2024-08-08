"""
作用: 从 stdout.log 绘制 loss 曲线
"""


import re
import matplotlib.pyplot as plt

def extract_data_from_log(log_file_path):
    general_loss_pattern = re.compile(
        r"Epoch = (\d+), Sum Loss = ([\d.]+), Avg Loss = ([\d.]+), CE Loss = ([\d.]+), Recon Loss = ([\d.]+), Avg Acc = ([\d.]+), Weighted Loss = ([\d.]+), Qtype_avg Loss = ([\d.]+)"
    )
    question_type_pattern = re.compile(
        r"Epoch (\d+), Question Type (\w): Total Loss = ([\d.]+), Count = (\d+), Avg Loss = ([\d.]+)"
    )

    general_losses = {}
    question_type_losses = {}

    with open(log_file_path, 'r') as file:
        for line in file:
            general_match = general_loss_pattern.search(line)
            if general_match:
                epoch = int(general_match.group(1))
                data = {
                    'Sum Loss': float(general_match.group(2)),
                    'Avg Loss': float(general_match.group(3)),
                    'CE Loss': float(general_match.group(4)),
                    'Recon Loss': float(general_match.group(5)),
                    'Avg Acc': float(general_match.group(6)),
                    'Weighted Loss': float(general_match.group(7)),
                    'Qtype_avg Loss': float(general_match.group(8))
                }
                general_losses[epoch] = data

            question_type_match = question_type_pattern.search(line)
            if question_type_match:
                epoch = int(question_type_match.group(1))
                q_type = question_type_match.group(2)
                data = {
                    'Total Loss': float(question_type_match.group(3)),
                    'Count': int(question_type_match.group(4)),
                    'Avg Loss': float(question_type_match.group(5))
                }
                if epoch not in question_type_losses:
                    question_type_losses[epoch] = {}
                question_type_losses[epoch][q_type] = data

    return general_losses, question_type_losses

def plot_data(general_losses, question_type_losses):
    epochs = list(general_losses.keys())
    sum_losses = [data['Sum Loss'] for data in general_losses.values()]
    avg_losses = [data['Avg Loss'] for data in general_losses.values()]
    ce_losses = [data['CE Loss'] for data in general_losses.values()]
    recon_losses = [data['Recon Loss'] for data in general_losses.values()]
    avg_accs = [data['Avg Acc'] for data in general_losses.values()]
    weighted_losses = [data['Weighted Loss'] for data in general_losses.values()]
    qtype_avg_losses = [data['Qtype_avg Loss'] for data in general_losses.values()]

    # Plot all metrics in one figure for comparison
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, sum_losses, label='Sum Loss', marker='o')
    plt.plot(epochs, avg_losses, label='Average Loss', marker='o')
    plt.plot(epochs, ce_losses, label='CE Loss', marker='o')
    plt.plot(epochs, recon_losses, label='Reconstruction Loss', marker='o')
    plt.plot(epochs, avg_accs, label='Average Accuracy', marker='o', color='green')
    plt.plot(epochs, weighted_losses, label='Weighted Loss', marker='o', color='purple')
    plt.plot(epochs, qtype_avg_losses, label='Qtype_avg Loss', marker='o', color='cyan')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Training Metrics Across Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Individual plots for each metric
    metrics = {
        'Sum Loss': sum_losses,
        'Average Loss': avg_losses,
        'CE Loss': ce_losses,
        'Reconstruction Loss': recon_losses,
        'Average Accuracy': avg_accs,
        'Weighted Loss': weighted_losses,
        'Qtype_avg Loss': qtype_avg_losses
    }

    for label, values in metrics.items():
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, values, label=label, marker='o', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel(label)
        plt.title(f'{label} Across Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
    

    # Plot question type losses
    plt.figure(figsize=(10, 5))
    for q_type in question_type_losses[0].keys():
        q_losses = [data[q_type]['Avg Loss'] for epoch, data in question_type_losses.items() if q_type in data]
        plt.plot(epochs, q_losses, label=f'Avg Loss {q_type}', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.title('Average Loss by Question Type Across Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# Usage remains the same
log_file_path = 'results/sutd-traffic/log/stdout.log'
general_losses, question_type_losses = extract_data_from_log(log_file_path)
plot_data(general_losses, question_type_losses)