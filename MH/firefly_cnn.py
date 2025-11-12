# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import datetime
import time
import os

############################################
# Configuration for scheme and directories #
############################################
esquema_num = 1
esquema_name = f'Esquema{esquema_num}testGPU'

# Condicionales para ajustar parameters de la metaheuristica en funci�n de esquema_num
if esquema_num == 1:
    fireflies_count = 10
    n_iter = 10
    alpha = 0.05
    beta0 = 0.8
    gamma = 20.0
elif esquema_num == 2:
    fireflies_count = 15
    n_iter = 20
    alpha = 0.1
    beta0 = 0.5
    gamma = 10.0
elif esquema_num == 3:
    fireflies_count = 20
    n_iter = 30
    alpha = 0.15
    beta0 = 0.3
    gamma = 5.0
elif esquema_num == 4:
    fireflies_count = 25
    n_iter = 40
    alpha = 0.25
    beta0 = 0.2
    gamma = 1.0

os.makedirs(esquema_name, exist_ok=True)
os.makedirs(os.path.join(esquema_name, 'Historial'), exist_ok=True)

#####################################################
# Function to train CNN model with Early Stopping
#####################################################
def train_cnn_model(num_conv_layers, base_filter_value, use_batch_norm, lr, batch_size, epochs, cs_iter, cnn_count, esquema_name):
    start_time = time.time()
    folder_name = os.path.join(esquema_name, f'ejecucion_cnn_iter_{cs_iter}_cnn_{cnn_count}')
    os.makedirs(folder_name, exist_ok=True)
    log_file_path = os.path.join(folder_name, 'training_log.txt')
    model_save_path = os.path.join(folder_name, 'best_model.pth')


    # Definir el dispositivo: GPU si está disponible, sino CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    class CustomCNN(nn.Module):
        def __init__(self, num_conv_layers, base_filter_value, use_batch_norm):
            super(CustomCNN, self).__init__()
            layers = []
            in_channels = 3
            current_filters = base_filter_value

            for i in range(num_conv_layers):
                layers.append(nn.Conv2d(in_channels, current_filters, kernel_size=3, padding=1))
                if use_batch_norm:
                    layers.append(nn.BatchNorm2d(current_filters))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                in_channels = current_filters
                current_filters *= 2

            self.features = nn.Sequential(*layers)
            self.fc1 = nn.Linear(in_channels * (64 // 2**num_conv_layers)**2, 1024)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(1024, 2)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # HPC dataset paths
    train_path = 'data_procesada/train'
    valid_path = 'data_procesada/valid'
    test_path = 'data_procesada/test'

    train_data = datasets.ImageFolder(train_path, transform=transform)
    valid_data = datasets.ImageFolder(valid_path, transform=transform)
    test_data = datasets.ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    model = CustomCNN(num_conv_layers, base_filter_value, use_batch_norm)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # Early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.0)

    train_loss_history, valid_loss_history, valid_accuracy_history = [], [], []
    y_true_all, y_pred_all = [], []

    best_valid_accuracy = 0
    with open(log_file_path, 'w') as log_file:
        log_file.write("Training Log\n")
        log_file.write("=============\n")
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * data.size(0)

            average_train_loss = total_train_loss / len(train_loader.dataset)
            train_loss_history.append(average_train_loss)

            model.eval()
            total_valid_loss, valid_correct, total_valid_samples = 0, 0, 0
            y_true_epoch, y_pred_epoch = [], []
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    total_valid_loss += loss.item() * data.size(0)
                    pred = output.argmax(dim=1, keepdim=True)
                    valid_correct += pred.eq(target.view_as(pred)).sum().item()
                    total_valid_samples += data.size(0)
                    y_true_epoch.extend(target.cpu().numpy())
                    y_pred_epoch.extend(pred.cpu().numpy())

            average_valid_loss = total_valid_loss / total_valid_samples
            valid_accuracy = 100. * valid_correct / total_valid_samples  # Accuracy in percentage
            valid_loss_history.append(average_valid_loss)
            valid_accuracy_history.append(valid_accuracy)
            y_true_all.extend(y_true_epoch)
            y_pred_all.extend(y_pred_epoch)

            # Guardar mejor modelo segun valid_accuracy
            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                torch.save(model.state_dict(), model_save_path)

            epoch_msg = (f'Epoch {epoch+1}/{epochs}, Training Loss: {average_train_loss:.4f}, '
                         f'Validation Loss: {average_valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%\n')
            log_file.write(epoch_msg)
            print(epoch_msg, end='')

            # Llamar al early stopping
            early_stopping(average_valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # Si early stopping ocurrio, cargar el mejor modelo guardado por early_stopping
        early_stopping.load_best_model(model)

        final_average_valid_loss = sum(valid_loss_history) / len(valid_loss_history)
        final_average_valid_accuracy = sum(valid_accuracy_history) / len(valid_accuracy_history)

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        conf_matrix = confusion_matrix(y_true_all, y_pred_all)
        recall = recall_score(y_true_all, y_pred_all, average='macro')
        auc_score = roc_auc_score(y_true_all, y_pred_all)
        f1 = f1_score(y_true_all, y_pred_all, average='macro')

        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp)

        end_time = time.time()
        total_execution_time = end_time - start_time
        hours, rem = divmod(total_execution_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

        final_msg = (f'\nFinal Results\n'
                     f'Final Average Validation Loss: {final_average_valid_loss:.4f}\n'
                     f'Final Average Validation Accuracy: {final_average_valid_accuracy:.2f}%\n'
                     f'Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1 Score: {f1:.4f}, AUC: {auc_score:.4f}\n'
                     f'Total Execution Time: {time_formatted} (hh:mm:ss)\n')
        log_file.write(final_msg)

        conf_matrix_msg = (f'\nConfusion Matrix:\n{conf_matrix}\n')
        log_file.write(conf_matrix_msg)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Real')
    plt.savefig(os.path.join(folder_name, 'confusion_matrix.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(valid_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(folder_name, 'loss_plot.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(valid_accuracy_history, label='Validation Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(folder_name, 'accuracy_plot.png'))
    plt.close()

    metrics_values = {'Recall': recall, 'Specificity': specificity, 'F1 Score': f1, 'AUC': auc_score}
    metric_names = list(metrics_values.keys())
    metric_values = list(metrics_values.values())

    plt.figure(figsize=(10, 5))
    pastel_colors = ['#aec6cf', '#ffb3ba', '#baffc9', '#ffdfba']
    bars = plt.bar(metric_names, metric_values, color=pastel_colors)
    plt.title('Final Metrics: Recall, Specificity, F1 Score, AUC')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}',
                 ha='center', va='bottom', fontsize=10, color='black')

    plt.savefig(os.path.join(folder_name, 'metrics_bar_plot.png'), bbox_inches='tight')
    plt.close()

    return final_average_valid_loss, final_average_valid_accuracy, recall, specificity, f1, auc_score, total_execution_time

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

##########################################################
# Function to evaluate fitness
##########################################################
def evaluate_fitness(firefly, device, cs_iter, cnn_count, esquema_name):
    loss, accuracy, recall, specificity, f1, auc, execution_time = train_cnn_model(
    num_conv_layers=firefly['num_layers'],
    base_filter_value=firefly['num_filters'],
    use_batch_norm=firefly['batch_norm'] == 'true',
    lr=firefly['lr'],
    batch_size=firefly['batch_size'],
    epochs=firefly['epochs'],
    cs_iter=cs_iter,
    cnn_count=cnn_count,
    esquema_name=esquema_name
    )
    return {
        'average_accuracy': accuracy,   # accuracy ya en %
        'loss': loss,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auc': auc,
        'execution_time': execution_time
    }

##############################################
# Euclidean distance function
##############################################
def euclidean_distance(firefly1, firefly2):
    distance = 0
    for key in firefly1:
        if isinstance(firefly1[key], str) or isinstance(firefly2[key], str):
            continue
        elif isinstance(firefly1[key], list):
            distance += sum((np.array(firefly1[key]) - np.array(firefly2[key]))**2)
        else:
            distance += (firefly1[key] - firefly2[key])**2
    return np.sqrt(distance)

##########################################
# Move firefly function
##########################################
def move_firefly(firefly_i, firefly_j, beta, alpha, param_ranges):
    valid = False
    while not valid:
        new_firefly = {}
        for key, value in firefly_i.items():
            if isinstance(param_ranges[key], tuple):
                move = beta * (firefly_j[key] - firefly_i[key]) + alpha * (np.random.random() - 0.5)
                new_value = firefly_i[key] + move
                low, high = param_ranges[key]
                new_firefly[key] = int(round(max(low, min(new_value, high)))) if key == "epochs" else max(low, min(new_value, high))

            elif isinstance(param_ranges[key], list) and isinstance(value, (int, float)):
                move = beta * (firefly_j[key] - firefly_i[key]) + alpha * (np.random.random() - 0.5)
                new_value = firefly_i[key] + move
                new_firefly[key] = min(param_ranges[key], key=lambda x: abs(x - new_value))

            elif isinstance(param_ranges[key], list) and isinstance(value, str):
                new_firefly[key] = firefly_j[key] if random.random() < 0.5 else firefly_i[key]

        valid = True
    return new_firefly

###########################################
# Initialize population
###########################################
def initialize_population(count, param_ranges):
    population = []
    for _ in range(count):
        firefly = {}
        for param, options in param_ranges.items():
            if isinstance(options, list):
                firefly[param] = random.choice(options)
            elif isinstance(options, tuple):
                firefly[param] = round(random.uniform(options[0], options[1]), 4)
                if param == "epochs":
                    firefly[param] = int(firefly[param])
        population.append(firefly)
    return population

#################################
# Firefly Algorithm main method #
#################################
def firefly_algorithm(iterations, fireflies_count, device, esquema_name):
    history_file_path = os.path.join(esquema_name, 'Historial', 'History_FA.txt')
    with open(history_file_path, 'w') as f:
        f.write("")

    best_firefly = None
    best_value = float('-inf')
    best_accuracy = 0.0
    best_loss = float('inf')
    best_time = 0.0
    best_recall = 0.0
    best_specificity = 0.0
    best_f1 = 0.0
    best_auc = 0.0
    iteration_the_best = 0
    cnn_the_best = 0

    # Initialize population and fitnesses
    population = initialize_population(fireflies_count, param_ranges)
    fitnesses = [evaluate_fitness(population[i], device, cs_iter=1, cnn_count=(i+1), esquema_name=esquema_name) for i in range(fireflies_count)]

    print("\nInitial Fireflies:")
    for index, firefly in enumerate(population):
        print(f"Firefly {index + 1}: {firefly}")

    best_accuracy_per_iteration = []
    best_loss_per_iteration = []
    best_execution_time_per_iteration = []
    best_recall_per_iteration = []
    best_specificity_per_iteration = []
    best_f1_per_iteration = []
    best_auc_per_iteration = []
    best_value_per_iteration = []

    for iteration in range(iterations):
        iter_best_value = float('-inf')
        iter_best_accuracy = 0.0
        iter_best_loss = float('inf')
        iter_best_time = 0.0
        iter_best_recall = 0.0
        iter_best_specificity = 0.0
        iter_best_f1 = 0.0
        iter_best_auc = 0.0
        iter_best_firefly = None

        for i in range(fireflies_count):
            for j in range(fireflies_count):
                if i != j and fitnesses[j]['average_accuracy'] > fitnesses[i]['average_accuracy']:
                    r = euclidean_distance(population[i], population[j])
                    beta = beta0 * np.exp(-gamma * r**2)
                    
                    population[i] = move_firefly(population[i], population[j], beta, alpha, param_ranges)
                    print("iteration", iteration + 1, "cnn", i + 1 + iteration * fireflies_count)
                    current_fitness = evaluate_fitness(population[i], device, cs_iter=iteration+1, cnn_count=(i+1+iteration*fireflies_count), esquema_name=esquema_name)
                    fitnesses[i] = current_fitness

                    # Update iteration best if current is better
                    if current_fitness['average_accuracy'] > iter_best_accuracy:
                        iter_best_value = current_fitness['average_accuracy']
                        iter_best_accuracy = current_fitness['average_accuracy']
                        iter_best_firefly = population[i]
                        iter_best_loss = current_fitness['loss']
                        iter_best_time = current_fitness['execution_time']
                        iter_best_recall = current_fitness['recall']
                        iter_best_specificity = current_fitness['specificity']
                        iter_best_f1 = current_fitness['f1']
                        iter_best_auc = current_fitness['auc']
                        iter_iteration_the_best = iteration + 1
                        iter_cnn_the_best = i + 1 + iteration * fireflies_count

        # Update global best if iteration best is better
        if iter_best_accuracy > best_accuracy:
            best_accuracy = iter_best_accuracy
            best_loss = iter_best_loss
            best_time = iter_best_time
            best_recall = iter_best_recall
            best_specificity = iter_best_specificity
            best_f1 = iter_best_f1
            best_auc = iter_best_auc
            best_value = iter_best_value
            best_firefly = iter_best_firefly
            best_iteration_the_best = iter_iteration_the_best   
            best_cnn_the_best = iter_cnn_the_best

        best_accuracy_per_iteration.append(iter_best_accuracy)
        best_loss_per_iteration.append(iter_best_loss)
        best_execution_time_per_iteration.append(iter_best_time)
        best_recall_per_iteration.append(iter_best_recall)
        best_specificity_per_iteration.append(iter_best_specificity)
        best_f1_per_iteration.append(iter_best_f1)
        best_auc_per_iteration.append(iter_best_auc)
        best_value_per_iteration.append(iter_best_value)

        # Write iteration results correctly
        with open(history_file_path, 'a') as f:
            f.write(f"Iteration {iteration + 1}:\n")
            f.write(f"Best value: {iter_best_value}\n")
            f.write(f"Best nest: {iter_best_firefly}\n")
            f.write(f"Best Average Accuracy: {iter_best_accuracy:.2f}%\n")  # Accuracy in %
            f.write(f"Best Average Loss: {iter_best_loss:.4f}\n")  # Loss as float
            f.write(f"Best Recall: {iter_best_recall:.4f}\n")
            f.write(f"Best Specificity: {iter_best_specificity:.4f}\n")
            f.write(f"Best F1 Score: {iter_best_f1:.4f}\n")
            f.write(f"Best AUC: {iter_best_auc:.4f}\n")
            f.write(f"Best Execution Time: {iter_best_time:.2f} seconds\n\n")

    # Final results
    with open(history_file_path, 'a') as f:
        f.write("*************************\n")
        f.write("Final Results\n")
        f.write("=============\n")
        f.write(f" iteration: {best_iteration_the_best}\n")
        f.write(f" CNN: {best_cnn_the_best}\n")
        f.write(f"Best overall value: {best_value}\n")
        f.write(f"Best overall nest: {best_firefly}\n")
        f.write(f"Best Overall Accuracy: {best_accuracy:.2f}%\n")  # Accuracy in %
        f.write(f"Best Overall Loss: {best_loss:.4f}\n")  # Loss as float
        f.write(f"Best Recall: {best_recall:.4f}\n")
        f.write(f"Best Specificity: {best_specificity:.4f}\n")
        f.write(f"Best F1 Score: {best_f1:.4f}\n")
        f.write(f"Best AUC: {best_auc:.4f}\n")
        f.write(f"Best Execution Time: {best_time:.2f} seconds\n")
        f.write("*************************\n")

    return (
        best_firefly, best_value, best_accuracy, best_loss, best_time,
        best_accuracy_per_iteration, best_loss_per_iteration, best_execution_time_per_iteration,
        best_value_per_iteration, best_recall_per_iteration, best_specificity_per_iteration,
        best_f1_per_iteration, best_auc_per_iteration
    )

######################################################
# Parameter ranges
######################################################
param_ranges = {
    "num_layers": [2, 3, 4, 5],
    "num_filters": [16, 32, 64], ## cambiar
    "batch_norm": ["true", "false"],
    "epochs": (20, 40),
    "batch_size": [16, 32, 64],
    "lr": (0.0001, 0.01)
}

#######################################
# Execution of the Firefly Algorithm  #
#######################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Working with GPU")
else:
    print("Working with CPU")

start_time_fa = time.time()
(best_nest, best_value, best_accuracy, best_loss, best_time,
 best_accuracy_per_iteration, best_loss_per_iteration,
 best_execution_time_per_iteration, best_value_per_iteration,
 best_recall_per_iteration, best_specificity_per_iteration,
 best_f1_per_iteration, best_auc_per_iteration) = firefly_algorithm(n_iter, fireflies_count, device=device, esquema_name=esquema_name)
end_time_fa = time.time()

print("\nBest Solution found:")
print(f"Firefly: {best_nest}, F(x) Value: {best_value}")
print(f"Best Average Accuracy: {best_accuracy:.2f}%")
print(f"Best Average Loss: {best_loss:.4f}")
print(f"Best Recall: {max(best_recall_per_iteration):.4f}")
print(f"Best Specificity: {max(best_specificity_per_iteration):.4f}")
print(f"Best F1 Score: {max(best_f1_per_iteration):.4f}")
print(f"Best AUC: {max(best_auc_per_iteration):.4f}")
print(f"Best Execution Time: {best_time:.2f} seconds")

total_time_cs = end_time_fa - start_time_fa
hours, rem = divmod(total_time_cs, 3600)
minutes, seconds = divmod(rem, 60)
time_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

history_file_path = os.path.join(esquema_name, 'Historial', 'History_FA.txt')
with open(history_file_path, 'a') as f:
    f.write("*************************\n")
    f.write(f"Tiempo total de ejecucion de CS: {time_formatted} (hh:mm:ss)\n")
    f.write("*************************\n")

# Objective function evolution
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_iter + 1), best_value_per_iteration, marker='o', linestyle='-', color='blue')
plt.title('Objective Function Best Value per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Best Value')
plt.grid(True)
plt.savefig(os.path.join(esquema_name, 'Historial', 'evolucion_funcion_objetivo.png'))
plt.close()

# Best accuracy per iteration
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_iter + 1), best_accuracy_per_iteration, marker='o', linestyle='-', color='green')
plt.title('Best Accuracy per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.savefig(os.path.join(esquema_name, 'Historial', 'mejor_precision_por_iteracion.png'))
plt.close()

# Best loss per iteration
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_iter + 1), best_loss_per_iteration, marker='o', linestyle='-', color='red')
plt.title('Best Loss per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(os.path.join(esquema_name, 'Historial', 'mejor_perdida_por_iteracion.png'))
plt.close()

# Best execution time per iteration
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_iter + 1), best_execution_time_per_iteration, marker='o', linestyle='-', color='purple')
plt.title('Best Execution Time per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.savefig(os.path.join(esquema_name, 'Historial', 'mejor_tiempo_por_iteracion.png'))
plt.close()

# Recall evolution
if len(best_recall_per_iteration) == n_iter:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_iter + 1), best_recall_per_iteration, marker='o', linestyle='-', color='orange')
    plt.title('Recall Evolution per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.savefig(os.path.join(esquema_name, 'Historial', 'evolucion_recall_por_iteracion.png'))
    plt.close()

# Specificity evolution
if len(best_specificity_per_iteration) == n_iter:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_iter + 1), best_specificity_per_iteration, marker='o', linestyle='-', color='cyan')
    plt.title('Specificity Evolution per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Specificity')
    plt.grid(True)
    plt.savefig(os.path.join(esquema_name, 'Historial', 'evolucion_specificity_por_iteracion.png'))
    plt.close()

# F1 Score evolution
if len(best_f1_per_iteration) == n_iter:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_iter + 1), best_f1_per_iteration, marker='o', linestyle='-', color='magenta')
    plt.title('F1 Score Evolution per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.savefig(os.path.join(esquema_name, 'Historial', 'evolucion_f1_score_por_iteracion.png'))
    plt.close()

# AUC evolution
if len(best_auc_per_iteration) == n_iter:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_iter + 1), best_auc_per_iteration, marker='o', linestyle='-', color='brown')
    plt.title('AUC Evolution per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.savefig(os.path.join(esquema_name, 'Historial', 'evolucion_auc_por_iteracion.png'))
    plt.close()

# Final metrics bar plot
final_best_metrics = [max(best_recall_per_iteration), max(best_specificity_per_iteration), max(best_f1_per_iteration), max(best_auc_per_iteration)]
metric_labels = ['Recall', 'Specificity', 'F1 Score', 'AUC']
plt.figure(figsize=(10, 6))
pastel_colors = ['#aec6cf', '#ffb3ba', '#baffc9', '#ffdfba']
bars = plt.bar(metric_labels, final_best_metrics, color=pastel_colors)
plt.title('Best Final Metrics Obtained')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}',
             ha='center', va='bottom', fontsize=10, color='black')

plt.savefig(os.path.join(esquema_name, 'Historial', 'mejores_metricas_finales.png'))
plt.close()
