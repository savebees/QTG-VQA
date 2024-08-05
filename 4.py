import re
import matplotlib.pyplot as plt

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
                epochs.append(epoch)

            accuracy_match = accuracy_pattern.search(line)
            if accuracy_match:
                q_type = accuracy_match.group(1)
                acc = float(accuracy_match.group(2))
                if q_type not in accuracies:
                    accuracies[q_type] = []
                accuracies[q_type].append(acc)

    return epochs, accuracies

def plot_accuracies(epochs, accuracies):
    plt.figure(figsize=(12, 6))
    for q_type, acc_list in accuracies.items():
        if len(acc_list) == len(epochs):  # Ensure we have a full series of data
            plt.plot(epochs, acc_list, label=f'Accuracy for {q_type}', marker='o')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Question Type Across Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
log_file_path = 'results/sutd-traffic/validation_log.txt'
epochs, accuracies = extract_accuracy_data(log_file_path)
plot_accuracies(epochs, accuracies)