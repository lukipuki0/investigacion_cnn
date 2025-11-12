#-*- coding: utf-8 -*-
#**

import numpy as np
from scipy.stats import norm
from scipy.special import gamma
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score, roc_curve

# Libraries for CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import datetime
import time
import os

# *************************        FUNCTION FOR CNN          **************************
esquema_num = 4 # Cambiar el n�mero para cambiar de esquema
esquema_name = f'PSO{esquema_num}'  

# seleccionar parametros de esquemas
if esquema_num == 1:
    S = 10 
    C1 = 1  
    C2 = 1 
    MAX_ITERATIONS = 10
    neighborhood_size = 2   
elif esquema_num == 2:
    S = 15
    C1 = 2
    C2 = 2 
    MAX_ITERATIONS = 20
    neighborhood_size = 2  
elif esquema_num == 3:
    S = 20
    C1 = 3
    C2 = 3 
    MAX_ITERATIONS = 30
    neighborhood_size = 2  
elif esquema_num == 4:
    S = 25
    C1 = 4
    C2 = 4 
    MAX_ITERATIONS = 40
    neighborhood_size = 2  


def train_cnn_model(num_conv_layers, base_filter_value, use_batch_norm, lr, batch_size, epochs, iter, cnn_count):
    start_time = time.time()  # start time
    now = datetime.datetime.now()
    formatted_date = now.strftime('%Y%m%d_%H%M%S')

    # Directorio donde se guardarán los resultados
    base_directory = f'{esquema_name}'
    folder_name = os.path.join(base_directory, f'ejecucion_cnn_iter_{iter}_cnn_{cnn_count}')
    os.makedirs(folder_name, exist_ok=True)
    log_file_path = os.path.join(folder_name, 'training_log.txt')
    model_save_path = os.path.join(folder_name, 'best_model.pth')

    
    print("Hiperpar�metros:")
    print(f"num_conv_layers: {num_conv_layers}")
    print(f"base_filter_value: {base_filter_value}")
    print(f"use_batch_norm: {use_batch_norm}")
    print(f"lr: {lr}")
    print(f"batch_size: {batch_size}")
    print(f"epochs: {epochs}")


    # Definir el dispositivo: GPU si está disponible, sino CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    class CustomCNN(nn.Module):
        def __init__(self, num_conv_layers, base_filter_value, use_batch_norm):
            super(CustomCNN, self).__init__()
            layers = []
            in_channels = 3  # Asumiendo imágenes RGB
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

    # Data loading y transformaciones
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

    # Crear el modelo y moverlo al dispositivo (GPU o CPU)
    model = CustomCNN(num_conv_layers, base_filter_value, use_batch_norm).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    train_loss_history, valid_loss_history, valid_accuracy_history = [], [], []
    y_true_all, y_pred_all = [], []

    best_valid_accuracy = 0
    with open(log_file_path, 'w') as log_file:
        log_file.write("Training Log\n")
        log_file.write("=============\n")
        print('\nResultados por epochs')
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            for data, target in train_loader:
                # Mover datos y etiquetas al dispositivo
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
            valid_accuracy = 100. * valid_correct / total_valid_samples
            valid_loss_history.append(average_valid_loss)
            valid_accuracy_history.append(valid_accuracy)
            y_true_all.extend(y_true_epoch)
            y_pred_all.extend(y_pred_epoch)

            # Guardar el mejor modelo
            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                torch.save(model.state_dict(), model_save_path)

            epoch_msg = (f'Epoch {epoch+1}/{epochs}, Training Loss: {average_train_loss:.4f}, Validation Loss: {average_valid_loss:.4f}, '
                        f'Validation Accuracy: {valid_accuracy:.2f}%\n')
            log_file.write(epoch_msg)
            print(epoch_msg)

        # Resultados finales
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

        # Escribir los resultados finales en el archivo
        final_msg = (f'\nFinal Results\n'
                     f'Final Average Validation Loss: {final_average_valid_loss:.4f}\n'
                     f'Final Average Validation Accuracy: {final_average_valid_accuracy:.2f}%\n'
                     f'Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1 Score: {f1:.4f}, AUC: {auc_score:.4f}\n'
                     f'Total Execution Time: {time_formatted} (hh:mm:ss)\n')
        log_file.write(final_msg)

        # Escribir la matriz de confusi�n en el archivo
        conf_matrix_msg = (f'\nConfusion Matrix:\n{conf_matrix}\n')
        log_file.write(conf_matrix_msg)
        print(final_msg)
        print(conf_matrix_msg)

    # Guardar gr�ficos
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Real')
    plt.savefig(os.path.join(folder_name, 'confusion_matrix.png'))
    #plt.show()  # Mostrar la matriz de confusi�n   **** Comentado para HPC ****
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

    # Se define paleta de colores para graficos de barra
    plt.figure(figsize=(10, 5))
    pastel_colors = ['#aec6cf', '#ffb3ba', '#baffc9', '#ffdfba']  # Soft pastel shades of blue, pink, green, and orange
    bars = plt.bar(metric_names, metric_values, color=pastel_colors)
    #bars = plt.bar(metric_names, metric_values, color=['orange', 'cyan', 'magenta', 'brown'])
    plt.title('Final Metrics: Recall, Specificity, F1 Score, AUC')
    plt.ylabel('Value')
    plt.ylim(0, 1)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom', fontsize=10, color='black')

    plt.savefig(os.path.join(folder_name, 'metrics_bar_plot.png'), bbox_inches='tight')
    #plt.show()  # Mostrar gr�fico de m�tricas finales   **** Comentado para HPC ****
    plt.close()

    return final_average_valid_loss, final_average_valid_accuracy, recall, specificity, f1, auc_score, total_execution_time, model


# ************************* FUNCION OBJETIVO *************************
def evaluar_cnn(hiperparametros, it, cnn_count):
    return train_cnn_model(
        num_conv_layers=hiperparametros['num_layers'],
        base_filter_value=hiperparametros['num_filters'],
        use_batch_norm=hiperparametros['batch_norm'],
        lr=hiperparametros['lr'],
        batch_size=hiperparametros['batch_size'],
        epochs=hiperparametros['epochs'],
        iter=it,
        cnn_count=cnn_count,
    )

# ************************* FUNCION PSO *************************

# Inicializacion de particulas
def initialize_particles(S, param_ranges):
    particles = []
    velocities = []
    p_best = []
    p_best_fitness = [float('-inf')] * S  # Inicializamos el fitness de las mejores posiciones personales como -infinito

    for i in range(S):
        particle = {}
        velocity = {}

        # Inicializar parámetros de cada partícula
        for param, options in param_ranges.items():
            if isinstance(options, list):  # Parámetros categóricos
                particle[param] = random.choice(options)
                velocity[param] = 0  # No es necesario para los categóricos
            elif isinstance(options, tuple):  # Rango numérico (por ejemplo, 'lr' o 'epochs')
                particle[param] = round(random.uniform(options[0], options[1]), 4)
                if param == "epochs":
                    particle[param] = int(particle[param])  # Aseguramos que 'epochs' sea un entero
                velocity[param] = random.uniform(-1, 1)  # Velocidad inicial aleatoria

        particles.append(particle)
        velocities.append(velocity)
        p_best.append(particle.copy())  # La mejor posición inicial es la partícula misma

    return particles, velocities, p_best, p_best_fitness

# Limitar valores dentro de los rangos
def limit_values(particle, velocity, param_ranges):
    for param in particle:
        if isinstance(param_ranges[param], tuple):  # Si es un rango numerico
            # Limitar la velocidad
            if velocity[param] > param_ranges[param][1]:
                velocity[param] = param_ranges[param][1]
            elif velocity[param] < param_ranges[param][0]:
                velocity[param] = param_ranges[param][0]

            # Actualizar la posicion (valor del hiperparametro)
            particle[param] += velocity[param]

            # Asegurar que la posicion (valor) este dentro de los limites
            if particle[param] > param_ranges[param][1]:
                particle[param] = param_ranges[param][1]
            elif particle[param] < param_ranges[param][0]:
                particle[param] = param_ranges[param][0]

            if param == 'lr':
                particle[param] = max(0.0001, min(particle[param], 0.01))  # Mantener lr dentro de [0.0001, 0.01]
            
            if param == 'epochs':
                particle[param] = max(20, min(particle[param], 40))  
            
            if param == 'num_filters':
                particle[param] = max(64, min(particle[param], 256))

            if param == 'num_layers':
                particle[param] = max(2,min(particle[param],5))

            if param == 'batch_size':
                particle[param] = max(64,min(particle[param],256))

        

    return particle, velocity

def get_neighborhood_best(particles, p_best, p_best_fitness, neighborhood_size, index):
    """
    Obtiene la mejor solución personal dentro del vecindario de una partícula.
    """
    n_particles = len(particles)
    neighbors = []

    # Definir los índices de las partículas vecinas considerando el vecindario circular
    for offset in range(-neighborhood_size // 2, neighborhood_size // 2 + 1):
        neighbor_index = (index + offset) % n_particles
        neighbors.append((p_best[neighbor_index], p_best_fitness[neighbor_index]))

    # Retornar el mejor vecino basado en el fitness
    best_neighbor = max(neighbors, key=lambda x: x[1])
    return best_neighbor[0]  # Retornamos los parámetros de la mejor solución del vecindario


def local_best_pso(S, param_ranges, C1, C2, MAX_ITERATIONS, neighborhood_size):
    # Inicializar partículas, velocidades y métricas
    particles, velocities, p_best, p_best_fitness = initialize_particles(S, param_ranges)
    metrics_history = [{'fitness': float('-inf'), 'parameters': None} for _ in range(S)]
    it = 0

    # Crear directorio para guardar logs
    pso_log_directory = './esquema1'
    os.makedirs(pso_log_directory, exist_ok=True)
    pso_log_file = os.path.join(pso_log_directory, 'pso_training_log_per_iteration.txt')

    while it < MAX_ITERATIONS:
        print(f"\n--- Iteración {it+1} ---\n")

        # Abrir archivo de log para escribir resultados
        with open(pso_log_file, 'a') as log_file:
            log_file.write(f"\n--- Iteración {it+1} ---\n")

            # Evaluar cada partícula y actualizar su mejor solución personal
            cnn_count= 1
            for i in range(S):
                print(i)
                loss, accuracy, recall, specificity, f1, auc_score, total_time, model_obj =  evaluar_cnn(particles[i], it, cnn_count)
                metrics_history[i] = {'fitness': accuracy, 'parameters': particles[i].copy()}
                
                cnn_count += 1
                # Comparar con el mejor personal
                if accuracy > p_best_fitness[i]:
                    p_best[i] = particles[i].copy()
                    p_best_fitness[i] = accuracy

            # Actualizar velocidades y posiciones basadas en el mejor vecino local
            for i in range(S):
                l_best = get_neighborhood_best(particles, p_best, p_best_fitness, neighborhood_size, i)

                for param in param_ranges:
                    if isinstance(param_ranges[param], tuple):  # Solo actualizar parámetros numéricos
                        velocities[i][param] = (velocities[i][param] +
                                                C1 * random.uniform(0, 1) * (p_best[i][param] - particles[i][param]) +
                                                C2 * random.uniform(0, 1) * (l_best[param] - particles[i][param]))
                particles[i], velocities[i] = limit_values(particles[i], velocities[i], param_ranges)

            # Registrar métricas de la iteración
            avg_fitness = np.mean([m['fitness'] for m in metrics_history])
            max_fitness = max([m['fitness'] for m in metrics_history])
            best_particle = max(metrics_history, key=lambda m: m['fitness'])

            log_msg = (f"Iteración {it+1}, Avg Fitness: {avg_fitness:.4f},\n "
                       f"Max Fitness: {max_fitness:.4f},\n "
                       f"Best Hyperparameters: {best_particle['parameters']}\n")
            log_file.write(log_msg)
            print(log_msg)

        # Incrementar iteración
        it += 1

    # Retornar la mejor solución global al final del proceso
    best_global = max(metrics_history, key=lambda m: m['fitness'])
    return best_global['parameters']


# Rangos de los hiperparametros de CNN
param_ranges = {
    "num_layers": [3, 4, 5],
    "num_filters": [64, 128, 256],
    "batch_norm": [True, False],
    "epochs": (20, 40),
    "batch_size": [64, 128, 256],
    "lr": (0.0001, 0.01)
}

best_hyperparams = local_best_pso(S, param_ranges, C1, C2, MAX_ITERATIONS, neighborhood_size)

print("Mejores hiperparametros encontrados:", best_hyperparams)