#-*- coding: utf-8 -*-
#**

import numpy as np
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

esquema_num = 4 # Cambiar el n�mero para cambiar de esquema
esquema_name = f'CSA{esquema_num}'  

# seleccionar parametros de esquemas
if esquema_num == 1:
    n_crows = 10     
    max_iter = 10
elif esquema_num == 2:
    n_crows = 15
    max_iter = 20
elif esquema_num == 3:
    n_crows = 20
    max_iter = 30
elif esquema_num == 4:
    n_crows = 25
    max_iter = 40

#************************************************************************************************************
#*********************************** FUNCION PARA CNN *******************************************************
def train_cnn_model(num_conv_layers, base_filter_value, use_batch_norm, lr, batch_size, epochs, iter, cnn_count):
    start_time = time.time()  # start time
    now = datetime.datetime.now()
    formatted_date = now.strftime('%Y%m%d_%H%M%S')

    # Directorio donde se guardar�n los resultados
    base_directory = f'{esquema_name}'
    folder_name = os.path.join(base_directory, f'ejecucion_cnn_iter_{iter}_cnn_{cnn_count}')
    os.makedirs(folder_name, exist_ok=True)
    log_file_path = os.path.join(folder_name, 'training_log.txt')
    model_save_path = os.path.join(folder_name, 'best_model.pth')
    #escribir los hiperpar�metros 


    print("Hiperpar�metros:")
    print(f"num_conv_layers: {num_conv_layers}")
    print(f"base_filter_value: {base_filter_value}")
    print(f"use_batch_norm: {use_batch_norm}")
    print(f"lr: {lr}")
    print(f"batch_size: {batch_size}")
    print(f"epochs: {epochs}")

    # Definir el dispositivo: GPU si est� disponible, sino CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    class CustomCNN(nn.Module):
        def __init__(self, num_conv_layers, base_filter_value, use_batch_norm):
            super(CustomCNN, self).__init__()
            layers = []
            in_channels = 3  # Asumiendo im�genes RGB
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

    train_path = '../dataset_procesado_img_cv/train'
    valid_path = '../dataset_procesado_img_cv/valid'
    test_path = '../dataset_procesado_img_cv/test'



    train_data = datasets.ImageFolder(train_path, transform=transform)
    valid_data = datasets.ImageFolder(valid_path, transform=transform)
    test_data = datasets.ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    # Crear el modelo y moverlo al dispositivo (GPU o CPU)
    model = CustomCNN(num_conv_layers, base_filter_value, use_batch_norm).to(device)
    model.to(device)
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
            valid_accuracy = valid_correct / total_valid_samples
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

        # Escribir la matriz de confusi n en el archivo
        conf_matrix_msg = (f'\nConfusion Matrix:\n{conf_matrix}\n')
        log_file.write(conf_matrix_msg)
        print(final_msg)
        print(conf_matrix_msg)

    # Guardar gr ficos
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Real')
    plt.savefig(os.path.join(folder_name, 'confusion_matrix.png'))
    #plt.show()  # Mostrar la matriz de confusi n   **** Comentado para HPC ****
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
    #plt.show()  # Mostrar gr fico de m tricas finales   **** Comentado para HPC ****
    plt.close()

    return final_average_valid_loss, final_average_valid_accuracy, recall, specificity, f1, auc_score, total_execution_time, model


#************************************************************************************************************
#**********************************  CSA  *********************************
def evaluar_cnn(hiperparametros, iter, cnn_count):
    """Evalua la CNN con los hiperparametros dados."""
    accuracy = train_cnn_model(
        num_conv_layers=hiperparametros['num_layers'],
        base_filter_value=hiperparametros['num_filters'],
        use_batch_norm=hiperparametros['batch_norm'],
        lr=hiperparametros['lr'],
        batch_size=hiperparametros['batch_size'],
        epochs=hiperparametros['epochs'],
        iter=iter,        # Argumento requerido
        cnn_count=cnn_count     # Argumento requerido
    )
    return accuracy

# **Definicion de los parametros del CSA**

param_ranges = {
    "num_layers": [3, 4, 5],
    "num_filters": [64, 128, 256],
    "batch_norm": [True, False],
    "epochs": (20, 40),
    "batch_size": [64, 128, 256],
    "lr": (0.0001, 0.01)
}


def inicializar_poblacion(n, param_ranges):
    """
    Inicializa la poblacion de cuervos con valores aleatorios dentro de los rangos especificados.
    """
    population = []
    for _ in range(n):
        individuo = {}
        for param, values in param_ranges.items():
            if isinstance(values, list):  # Valores discretos o categ ricos
                individuo[param] = random.choice(values)
            elif isinstance(values, tuple):  # Valores continuos
                valor = round(random.uniform(values[0], values[1]), 4)
                if param == "epochs":  # Asegarate de que sea entero si es 'epochs'
                    valor = int(valor)
                individuo[param] = valor
        population.append(individuo)
    return population

def traducir_a_vector(individuo, param_ranges):
    """
    Convierte un diccionario de hiperparametros en un vector numerico.
    """
    vector = []
    for param, values in param_ranges.items():
        if isinstance(values, list):  #  ndice del valor discreto en la lista
            vector.append(values.index(individuo[param]))
        elif isinstance(values, tuple):  # Normalizaci n dentro del rango
            min_val, max_val = values
            vector.append((individuo[param] - min_val) / (max_val - min_val))
    return np.array(vector)

def traducir_a_diccionario(vector, param_ranges):
    """
    Convierte un vector numerico a un diccionario de hiperparametros.
    """
    print(vector)
    individuo = {}
    idx = 0       
    for param, values in param_ranges.items():
        if isinstance(values, list):  # Mapea al indice correspondiente
            indice_valido = min(max(int(round(vector[idx])), 0), len(values) - 1)
            individuo[param] = values[indice_valido]
        elif isinstance(values, tuple):  # Escala al rango original
            min_val, max_val = values
            individuo[param] = min_val + vector[idx] * (max_val - min_val)

            # Corregido: Asegurar que el learning rate (lr) nunca sea menor a 0.0001
            if param == "lr":
                individuo[param] = max(0.0001, individuo[param]) 
            
            # Corregido: Asegurar que 'epochs' sea un entero
            if param == "epochs":
                individuo[param] = int(individuo[param])  

            # Asegurar que el valor de lr esté en el rango permitido
            if param == 'lr':
                individuo[param] = max(0.0001, min(individuo[param], 0.01))  # Mantener lr dentro de [0.0001, 0.01]
            
            if param == 'epochs':
                individuo[param] = max(20, min(individuo[param], 40))  
                    
            if param == 'num_filters':
                individuo[param] = max(64, min(individuo[param], 256))

            if param == 'num_layers':
                individuo[param] = max(2,min(individuo[param],5))

            if param == 'batch_size':
                individuo[param] = max(64,min(individuo[param],256))

        idx += 1
    return individuo

def crow_search_algorithm_cnn(n, max_iter, param_ranges, AP, FL):
    """
    Implementa el CSA para optimizar los hiperparAmetros de una CNN.
    """
    # Inicializacion
    population = inicializar_poblacion(n, param_ranges)
    memory = [traducir_a_vector(ind, param_ranges) for ind in population]
    fit_mem = [evaluar_cnn(ind, iter=0, cnn_count=i)[0] for i, ind in enumerate(population)]
    history = []

    # L mites del espacio normalizado
    l, u = 0, 1
    dim = len(param_ranges)

    for t in range(max_iter):
        print(f"iteracion {t + 1}/{max_iter}")
        x = np.array(memory)
        xnew = np.empty_like(x)
        num = np.array([random.randint(0, n - 1) for _ in range(n)])

        for i in range(n):
            
            if random.random() > AP:  # Sigue al objetivo en memoria
                for j in range(dim):
                    xnew[i, j] = x[i, j] + FL * random.random() * (x[num[i], j] - x[i, j])
            else:  # Exploracion aleatoria
                for j in range(dim):
                    xnew[i, j] = l + (u - l) * random.random()

        # Evaluar nuevas posiciones
        new_population = [traducir_a_diccionario(vec, param_ranges) for vec in xnew]
        fit_new = [evaluar_cnn(ind, t, cnn_count=i)[0] for i, ind in enumerate(new_population)]

        # Actualizar memoria
        for i in range(n):
            if fit_new[i] > fit_mem[i]:
                memory[i] = xnew[i]
                fit_mem[i] = fit_new[i]

        # Guardar el mejor fitness de esta iteracion
        best_fitness = max(fit_mem)
        history.append(best_fitness)
        print(f"Mejor precision en esta iteracion: {best_fitness:.4f}")

    # Mejor solucion
    best_idx = fit_mem.index(max(fit_mem))
    best_hyperparams = traducir_a_diccionario(memory[best_idx], param_ranges)
    return best_hyperparams, history


# Ejecutar CSA para la CNN
best_hyperparams, history = crow_search_algorithm_cnn(
    n=n_crows,
    max_iter=max_iter,
    param_ranges=param_ranges,
    AP=0.1,
    FL=2.0
)

print("\nMejores hiperparametros encontrados:")
print(best_hyperparams)