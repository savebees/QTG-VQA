"""
作用: 从 validation_log.txt 绘制 acc 曲线（高亮最好的epoch）
"""

import re
import matplotlib.pyplot as plt
import numpy as np

def extract_accuracy_data(log_file_path):
    accuracy_pattern = re.compile(
        r"INFO:root:Accuracy for (\w): ([\d.]+)"
    )
    epoch_pattern = re.compile(
        r"INFO:root:Epoch (\d+) Validation Accuracy: ([\d.]+)"
    )

    accuracies = {}
    epochs = []

    with open(log_file_path, 'r') as file:
        for line in file:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                if epoch not in epochs:
                    epochs.append(epoch)

            accuracy_match = accuracy_pattern.search(line)
            if accuracy_match:
                q_type = accuracy_match.group(1)
                acc = float(accuracy_match.group(2))
                if q_type not in accuracies:
                    accuracies[q_type] = []
                accuracies[q_type].append(acc)

    return epochs, accuracies

def plot_accuracies(epochs, accuracies, window_size=5):
    plt.figure(figsize=(12, 6))
    best_epochs = {}

    for q_type, acc_list in accuracies.items():
        if len(acc_list) == len(epochs):  # Ensure we have a full series of data
            # Calculate moving average
            moving_average = np.convolve(acc_list, np.ones(window_size)/window_size, mode='valid')
            extended_epochs = epochs[window_size-1:]  # Adjust epochs for the size of the moving window

            # Find the best epoch and accuracy
            best_index = np.argmax(moving_average)
            best_epochs[q_type] = (extended_epochs[best_index], moving_average[best_index])

            plt.plot(extended_epochs, moving_average, label=f'Accuracy for {q_type}', marker='o')

            # Highlight the best epoch for each question type
            plt.scatter(*best_epochs[q_type], color='red', s=100, marker='*', zorder=5)
            plt.text(best_epochs[q_type][0], best_epochs[q_type][1], f' {best_epochs[q_type][0]}', verticalalignment='bottom')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Question Type Across Epochs (Moving Average)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
log_file_path = 'results/sutd-traffic/validation_log.txt'
epochs, accuracies = extract_accuracy_data(log_file_path)
plot_accuracies(epochs, accuracies)