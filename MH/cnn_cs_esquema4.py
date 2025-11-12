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

esquema_num = 2  # Cambiar el n�mero para cambiar de esquema

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


#*******************************************************************************************************************************************************
#********************************************************FUNCION     V U E L O     D E     L E V Y ( Es utilizada por CS) *******************************
#Adaptaci�n para cada tipo de par�metro (entero, flotante, booleano)
def levy_flight(nido, alpha, lambda_, param_ranges):
    nuevo_nido = nido.copy()
    d = len(param_ranges)
    sigma = (math.gamma(1 + lambda_) * np.sin(np.pi * lambda_ / 2) /
             (math.gamma((1 + lambda_) / 2) * lambda_ * 2 ** ((lambda_ - 1) / 2))) ** (1 / lambda_)
    u = norm.rvs(size=d) * sigma
    v = norm.rvs(size=d)
    step = u / (np.abs(v) ** (1 / lambda_))

    for i, param in enumerate(param_ranges.keys()):
        if param in nido:
            if isinstance(param_ranges[param], list):
                if isinstance(nido[param], int):  # Par�metro entero
                    nuevo_nido[param] = int(nido[param] + alpha * step[i])
                    # Asegurar que est� dentro del rango permitido
                    nuevo_nido[param] = max(min(nuevo_nido[param], max(param_ranges[param])), min(param_ranges[param]))
                elif isinstance(nido[param], str):  # Par�metro booleano
                    if random.random() < 0.5:
                        nuevo_nido[param] = 'true' if nido[param] == 'false' else 'false'
                elif param == "batch_size":
                    # Seleccionar aleatoriamente un valor de batch_size de los valores permitidos
                    nuevo_nido[param] = random.choice(param_ranges[param])
            elif isinstance(param_ranges[param], tuple):  # Par�metros por rango
                nuevo_nido[param] = nido[param] + alpha * step[i]
                # Asegurar que est� dentro del rango permitido
                nuevo_nido[param] = max(min(nuevo_nido[param], param_ranges[param][1]), param_ranges[param][0])
                nuevo_nido[param] = round(nuevo_nido[param], 4)
                if param == "epochs":
                    nuevo_nido[param] = int(nuevo_nido[param])
    return nuevo_nido


#**********************************************************************************************************************
#******************************************************** C U C K O O      S E A R C H *******************************
#Funci�n algoritmo Cuckoo Search
def cuckoo_search(nests, n_iter, alpha, lambda_, pa):
    # Initialize variables to track history and best values
    history = []
    best_nest = None
    best_value = float('-inf')  # Initialize to negative infinity to maximize accuracy
    best_accuracy = 0
    best_loss = float('inf')  # Initialize with a high value to minimize loss
    best_time = 0
    best_recall = 0
    best_specificity = 0
    best_f1 = 0
    best_auc = 0
    best_model = None  # Guardar el mejor modelo de todas las iteraciones
    iteration_the_best = 0
    cnn_the_best = 0
    #best_model_save_path = '/content/drive/MyDrive/Colab Notebooks/{esquema_name}/best_model.pth'    *****comentado para HPC****
    best_model_save_path = f'{esquema_name}/History/best_model.pth'
   
    history_file_path = os.path.join(f'{esquema_name}/History/History_CS.txt')
    #history_file_path = os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'History_CS.txt')   **** Comentado para HPC****
    # Asegurar que el directorio de historial existe, de lo contrario se crea
    os.makedirs(os.path.dirname(history_file_path), exist_ok=True)

    # Ensure the history file exists and is empty
    with open(history_file_path, 'w') as f:
        f.write("")

    # Lists to store the best metrics per iteration
    best_accuracy_per_iteration = []
    best_loss_per_iteration = []
    best_execution_time_per_iteration = []
    best_recall_per_iteration = []
    best_specificity_per_iteration = []
    best_f1_per_iteration = []
    best_auc_per_iteration = []
    best_value_per_iteration = []

    with open(history_file_path, 'w') as f:
        f.write("*************************")
        f.write("\nParametros de la ejecucion")
        f.write(f"\nNumero de lobos: {n_nest}")
        f.write(f"\nNumero de iteraciones: {n_iter}")
        f.write("\n*************************\n")
        f.write("pesos de la funcion objetivo\n")
        f.write(f"alpha: {alpha}\n")
        f.write(f"lambda_: {lambda_}\n")
        f.write(f"pa: {pa}\n")
        f.write("")

    # Initialize counters for the CNN executions within this iteration
    

    for i in range(n_iter):
        print("\n**********************************************  ITERATION ", i + 1, " **********************************************")
        print("\nCurrent population nests:")
        for nest in nests:
            print(nest)
        cnn_count = 1
        # Initialize values for tracking the best results in the current iteration
        best_iteration_value = float('-inf')
        best_iteration_accuracy = 0
        best_iteration_loss = float('inf')
        best_iteration_time = 0
        best_iteration_recall = 0
        best_iteration_specificity = 0
        best_iteration_f1 = 0
        best_iteration_auc = 0

        for idx in range(len(nests)):
            current_nest = nests[idx].copy()

            # Dentro de la funci�n cuckoo_search
            final_average_valid_loss, final_average_valid_accuracy, recall, specificity, f1, auc_score, total_execution_time, trained_model = train_cnn_model(
                num_conv_layers=current_nest['num_layers'],
                base_filter_value=current_nest['num_filters'],
                use_batch_norm=current_nest['batch_norm'],
                lr=current_nest['lr'],
                batch_size=current_nest['batch_size'],
                epochs=current_nest['epochs'],
                iter=i + 1,            # N�mero de iteraci�n de CS
                cnn_count=cnn_count        # N�mero correlativo dentro de la iteraci�n
            )
            cnn_count += 1

            current_value = final_average_valid_accuracy

            # Update the iteration's best values if there's an improvement
            if current_value > best_iteration_value:
                best_iteration_value = current_value
                best_iteration_accuracy = final_average_valid_accuracy
                best_iteration_loss = final_average_valid_loss
                best_iteration_time = total_execution_time
                best_iteration_recall = recall
                best_iteration_specificity = specificity
                best_iteration_f1 = f1
                best_iteration_auc = auc_score

            # Update the global best values if there's an improvement
            if current_value > best_value:
                best_value = current_value
                best_nest = current_nest.copy()
                best_accuracy = final_average_valid_accuracy
                best_loss = final_average_valid_loss
                best_time = total_execution_time
                best_recall = recall
                best_specificity = specificity
                best_f1 = f1
                best_auc = auc_score
                best_model = trained_model  # Guardar el mejor modelo
                cnn_the_best = cnn_count
                iteration_the_best = i + 1
            # Generate a new nest using L�vy flight and evaluate it
            if random.random() < pa:
                new_nest = levy_flight(current_nest, alpha, lambda_, param_ranges)
                final_average_valid_loss, final_average_valid_accuracy, recall, specificity, f1, auc_score, total_execution_time, trained_model = train_cnn_model(
                    num_conv_layers=new_nest['num_layers'],
                    base_filter_value=new_nest['num_filters'],
                    use_batch_norm=new_nest['batch_norm'],
                    lr=new_nest['lr'],
                    batch_size=new_nest['batch_size'],
                    epochs=new_nest['epochs'],
                    iter=i + 1,            # N�mero de iteraci�n de CS
                    cnn_count=cnn_count    # Actualizamos el contador de CNN dentro de la iteraci�n
                )
                cnn_count += 1
                new_value = final_average_valid_accuracy

                # Check if the new nest is better and update the current nest accordingly
                if new_value > current_value:
                    nests[idx] = new_nest

                    # Update the iteration's best values if there's an improvement
                    if new_value > best_iteration_value:
                        best_iteration_value = new_value
                        best_iteration_accuracy = final_average_valid_accuracy
                        best_iteration_loss = final_average_valid_loss
                        best_iteration_time = total_execution_time
                        best_iteration_recall = recall
                        best_iteration_specificity = specificity
                        best_iteration_f1 = f1
                        best_iteration_auc = auc_score

                    # Update the global best values if there's an improvement
                    if new_value > best_value:
                        best_value = new_value
                        best_nest = new_nest.copy()
                        best_accuracy = final_average_valid_accuracy
                        best_loss = final_average_valid_loss
                        best_time = total_execution_time
                        best_recall = recall
                        best_specificity = specificity
                        best_f1 = f1
                        best_auc = auc_score
                        best_model = trained_model  # Guardar el mejor modelo
                        cnn_the_best = cnn_count
                        iteration_the_best = i + 1
                

        # Append the best values of the current iteration to the corresponding lists
        best_accuracy_per_iteration.append(best_iteration_accuracy)
        best_loss_per_iteration.append(best_iteration_loss)
        best_execution_time_per_iteration.append(best_iteration_time)
        best_recall_per_iteration.append(best_iteration_recall)
        best_specificity_per_iteration.append(best_iteration_specificity)
        best_f1_per_iteration.append(best_iteration_f1)
        best_auc_per_iteration.append(best_iteration_auc)
        best_value_per_iteration.append(best_iteration_value)

        # Log the iteration results
        with open(history_file_path, 'a') as f:
            f.write(f"\nIteration {i + 1}:\n")
            f.write(f"Best value: {best_iteration_value}\n")
            f.write(f"Best nest: {best_nest}\n")
            f.write(f"Best Average Accuracy: {best_iteration_accuracy:.2f}%\n")
            f.write(f"Best Average Loss: {best_iteration_loss:.4f}\n")
            f.write(f"Best Recall: {best_iteration_recall:.4f}\n")
            f.write(f"Best Specificity: {best_iteration_specificity:.4f}\n")
            f.write(f"Best F1 Score: {best_iteration_f1:.4f}\n")
            f.write(f"Best AUC: {best_iteration_auc:.4f}\n")
            f.write(f"Best Execution Time: {best_iteration_time:.2f} seconds\n")
            f.write("\n")

        print("\nBest value for this iteration:", best_iteration_value)
        print("Best nest in this iteration:", best_nest)
        print(f"Best Average Accuracy: {best_iteration_accuracy:.2f}%")
        print(f"Best Average Loss: {best_iteration_loss:.4f}")
        print(f"Best Recall: {best_iteration_recall:.4f}")
        print(f"Best Specificity: {best_iteration_specificity:.4f}")
        print(f"Best F1 Score: {best_iteration_f1:.4f}")
        print(f"Best AUC: {best_iteration_auc:.4f}")
        print(f"Best Execution Time: {best_iteration_time:.2f} seconds")

    # Write final results after completing all iterations
    with open(history_file_path, 'a') as f:
        f.write("*************************\n")
        f.write("Final Results\n")
        f.write(f"iteracion: {iteration_the_best}\n")
        f.write(f"cnn: {cnn_the_best}\n")
        f.write(f"Best overall value: {best_value}\n")
        f.write(f"Best overall nest: {best_nest}\n")
        f.write(f"Best Overall Accuracy: {best_accuracy:.2f}%\n")
        f.write(f"Best Overall Loss: {best_loss:.4f}\n")
        f.write(f"Best Recall: {best_recall:.4f}\n")
        f.write(f"Best Specificity: {best_specificity:.4f}\n")
        f.write(f"Best F1 Score: {best_f1:.4f}\n")
        f.write(f"Best AUC: {best_auc:.4f}\n")
        f.write(f"Best Execution Time: {best_time:.2f} seconds\n")
        f.write("*************************\n")

    # Guardar el mejor modelo en la carpeta ra�z
    torch.save(best_model.state_dict(), best_model_save_path)
    print(f"El mejor modelo de todas las iteraciones ha sido guardado en: {best_model_save_path}")

    # Check if all metric lists have the correct length before plotting
    if len(best_recall_per_iteration) == n_iter:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_iter + 1), best_recall_per_iteration, marker='o', linestyle='-', color='orange')
        plt.title('Evolucion del Recall por Iteracion')
        plt.xlabel('Iteracion')
        plt.ylabel('Recall')
        plt.grid(True)
        plt.savefig(os.path.join(f'{esquema_name}/History', 'recall_evolution_per_iteration.png'))
        #plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'recall_evolution_per_iteration.png'))   **** Comentado para HPC ****
        plt.close()

    # Repeat similar checks for other metrics
    if len(best_specificity_per_iteration) == n_iter:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_iter + 1), best_specificity_per_iteration, marker='o', linestyle='-', color='cyan')
        plt.title('Evolucion de Specificity por Iteracion')
        plt.xlabel('Iteracion')
        plt.ylabel('Specificity')
        plt.grid(True)
        plt.savefig(os.path.join(f'{esquema_name}/History', 'specificity_evolution_per_iteration.png'))
        #plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'specificity_evolution_per_iteration.png'))
        plt.close()

    if len(best_f1_per_iteration) == n_iter:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_iter + 1), best_f1_per_iteration, marker='o', linestyle='-', color='magenta')
        plt.title('Evolucion del F1 Score por Iteracion')
        plt.xlabel('Iteracion')
        plt.ylabel('F1 Score')
        plt.grid(True)
        plt.savefig(os.path.join(f'{esquema_name}/History', 'f1_score_evolution_per_iteration.png'))
        #plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'f1_score_evolution_per_iteration.png'))
        plt.close()

    if len(best_auc_per_iteration) == n_iter:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_iter + 1), best_auc_per_iteration, marker='o', linestyle='-', color='brown')
        plt.title('Evolucion del AUC por Iteracion')
        plt.xlabel('Iteracion')
        plt.ylabel('AUC')
        plt.grid(True)
        plt.savefig(os.path.join(f'{esquema_name}/History', 'auc_evolution_per_iteration.png'))
        #plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'auc_evolution_per_iteration.png'))
        plt.close()

    return best_nest, best_value, best_accuracy, best_loss, best_time, pd.DataFrame(history, columns=['Iteration', 'Nest Index', 'Current Nest', 'F(x) Current Value',
                                                    'F(x) New Value', 'New Nest', 'Status', 'Accuracy',
                                                    'Loss', 'Recall', 'Specificity', 'F1 Score', 'AUC', 'Execution Time']), best_accuracy_per_iteration, best_loss_per_iteration, best_execution_time_per_iteration, best_value_per_iteration, best_recall, best_specificity, best_f1, best_auc


#�******************************************************************** I N I C I O     A L G O R I T M O ****************************************************************************
#************************************************************************************************************************************************************
# ******* Par�metros y ejecuci�n del algoritmo CS por Esquema**********


esquema_name = f'Esquema_cs{esquema_num}'  # Nombre de la carpeta de salida basado en esquema_num

# Condicionales para ajustar par�metros de la metaheuristica en funci�n de esquema_num
if esquema_num == 1:
    n_nest = 10
    n_iter = 10
    pa = 0.2
    alpha = 0.05
    lambda_ = 1.0
elif esquema_num == 2:
    n_nest = 15
    n_iter = 20
    pa = 0.22
    alpha = 0.03
    lambda_ = 1.2
elif esquema_num == 3:
    n_nest = 20
    n_iter = 30
    pa = 0.23
    alpha = 0.02
    lambda_ = 1.3
elif esquema_num == 4:
    n_nest = 25
    n_iter = 40
    pa = 0.25
    alpha = 0.01
    lambda_ = 1.5


# Imprimir los valores de los par�metros
print(f"Valores de configuracion para esquema_num = {esquema_num}:")
print(f"n_nest = {n_nest}, n_iter = {n_iter}, pa = {pa}, alpha = {alpha}, lambda_ = {lambda_}")


# *****Rangos de los hiperparametros de CNN*******
param_ranges = {
    "num_layers": [2, 3, 4, 5],
    "num_filters": [32, 64, 128, 256],
    "batch_norm": ["true", "false"],
    "epochs": (20, 50),
    "batch_size": [16, 32, 64],
    "lr": (0.0001, 0.01)
}

# Inicializar poblaci�n de nidos
def generar_nido_aleatorio(param_ranges):
    nido = {}
    for param, options in param_ranges.items():
        if isinstance(options, list):  # Si las opciones son una lista
            nido[param] = random.choice(options)  # Selecciona un valor aleatorio de la lista
        elif isinstance(options, tuple):  # Para rangos num�ricos, como 'lr' y epochs
            nido[param] = round(random.uniform(options[0], options[1]), 4)  # Genera un valor flotante aleatorio para lr, redondeado en 4 decimales
            if param == "epochs":
                nido[param] = int(nido[param])  # Asegurar que epochs sea un entero
    return nido

nests = [generar_nido_aleatorio(param_ranges) for _ in range(n_nest)]

# Mostrar nidos generados
print("\nNidos iniciales")
for nido in nests:
    print(nido)

# Ejecuci�n del algoritmo CS y obtenci�n de resultados
start_time_cs = time.time()  # start time

# Ahora viene la llamada de CS
best_nest, best_value, best_accuracy, best_loss, best_time, history, best_accuracy_per_iteration, best_loss_per_iteration, best_execution_time_per_iteration, best_value_per_iteration, best_recall, best_specificity, best_f1, best_auc = cuckoo_search(nests, n_iter, alpha, lambda_, pa)
end_time_cs = time.time()

# Mostrar el historial de iteraciones y la mejor soluci�n encontrada
print("\nHistory de iteraciones:")
print(history.to_string(index=False))
print("\nMejor solucion encontrada:")
print(f"Nido: {best_nest}, Valor de F(x): {best_value}")
print(f"Best Average Accuracy: {best_accuracy:.2f}%")
print(f"Best Average Loss: {best_loss:.4f}")
print(f"Best Recall: {best_recall:.4f}")
print(f"Best Specificity: {best_specificity:.4f}")
print(f"Best F1 Score: {best_f1:.4f}")
print(f"Best AUC: {best_auc:.4f}")
print(f"Best Execution Time: {best_time:.2f} seconds")

# Tiempo total CS
total_time_cs = end_time_cs - start_time_cs

# Convert tiempo total de ejecuci�n en horas, minutos, y segundos
hours, rem = divmod(total_time_cs, 3600)
minutes, seconds = divmod(rem, 60)
time_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
print('Total Total de ejecucion', time_formatted, '(hh:mm:ss)\n')

# Guardar el tiempo total de CS en el archivo de historial
history_file_path = os.path.join(f'{esquema_name}/History/History_CS.txt')
#history_file_path = os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'History_CS.txt')   ***No para HPC ***
with open(history_file_path, 'a') as f:
    f.write("*************************\n")
    f.write(f"Tiempo total de ejecucion de CS: {time_formatted} (hh:mm:ss)\n")
    f.write("*************************\n")

# Evoluci�n de la funci�n objetivo
# Guardar el gr�fico de la funci�n objetivo
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_iter + 1), best_value_per_iteration, marker='o', linestyle='-', color='blue')
plt.title('Valor de la Funcion Objetivo por Iteracion')
plt.xlabel('Iteracion')
plt.ylabel('Mejor Valor de la Funcion Objetivo')
plt.grid(True)
plt.savefig(os.path.join(f'{esquema_name}/History', 'objective_function_evolution.png'))
#plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'objective_function_evolution.png'))  #**No para HPC ***
plt.close()

# Guardar el gr�fico de la mejor Accuracy por iteraci�n
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_iter + 1), best_accuracy_per_iteration, marker='o', linestyle='-', color='green')
plt.title('Mejor Accuracy por Iteracion')
plt.xlabel('Iteracion')
plt.ylabel('Mejor Accuracy (%)')
plt.grid(True)
plt.savefig(os.path.join(f'{esquema_name}/History', 'best_precision_per_iteration.png'))
#plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'best_precision_per_iteration.png'))  #**No para HPC ***
plt.close()

# Guardar el gr�fico de la mejor p�rdida por iteraci�n
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_iter + 1), best_loss_per_iteration, marker='o', linestyle='-', color='red')
plt.title('Mejor Perdida por Iteracion')
plt.xlabel('Iteracion')
plt.ylabel('Mejor Perdida')
plt.grid(True)
plt.savefig(os.path.join(f'{esquema_name}/History', 'best_loss_per_iteration.png'))
#plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'best_loss_per_iteration.png')) #**No para HPC ***
plt.close()

# Guardar el gr�fico del mejor tiempo de ejecuci�n por iteraci�n
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_iter + 1), best_execution_time_per_iteration, marker='o', linestyle='-', color='purple')
plt.title('Mejor Tiempo de Ejecucion por Iteracion')
plt.xlabel('Iteracion')
plt.ylabel('Mejor Tiempo de Ejecucion (segundos)')
plt.grid(True)
plt.savefig(os.path.join(f'{esquema_name}/History', 'best_time_per_iteration.png'))
#plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'best_time_per_iteration.png'))    #**No para HPC ***
plt.close()

# Generar gr�fico final de las m�tricas Recall, Specificity, F1 y AUC con los mejores valores
best_metric_values = [best_recall, best_specificity, best_f1, best_auc]
metric_labels = ['Recall', 'Specificity', 'F1 Score', 'AUC']
plt.figure(figsize=(10, 6))
bars=plt.bar(metric_labels, best_metric_values, color=['orange', 'cyan', 'magenta', 'brown'])
plt.title('Mejores metricas obtenidas al final de la ejecucion')
plt.xlabel('Metrica')
plt.ylabel('Valor')
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom', fontsize=10, color='black')

plt.savefig(os.path.join(f'{esquema_name}/History', 'best_final_metrics.png'))
#plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'best_final_metrics.png')) #**No para HPC ***
#plt.show()  #**No para HPC ***
plt.close()