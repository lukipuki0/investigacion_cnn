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

esquema_num = 1  
esquema_name = f'Test{esquema_num}'  

# seleccionar parametros de esquemas
if esquema_num == 1:
    nwolves = 10     
    maxIter = 10
elif esquema_num == 2:
    nwolves = 15
    maxIter = 20
elif esquema_num == 3:
    nwolves = 20
    maxIter = 30
elif esquema_num == 4:
    nwolves = 25
    maxIter = 40
   
param_ranges = {
    "num_layers": [3, 4, 5],
    "num_filters": [64, 128, 256],
    "batch_norm": [True, False],
    "epochs": (20, 40),
    "batch_size": [64, 128, 256],
    "lr": (0.0001, 0.01)
}   


#************************************************************************************************************
#*********************************** FUNCION PARA CNN *******************************************************
def train_cnn_model(num_conv_layers, base_filter_value, use_batch_norm, lr, batch_size, epochs, iter, cnn_count, early_stopping_patience=5):
    

    start_time = time.time()
    now = datetime.datetime.now()
    formatted_date = now.strftime('%Y%m%d_%H%M%S')

    # Directory where CNN executions will be saved with the CS iteration number and number correlativo dentro de la iteraci n
    #base_directory = f'{esquema_name}'  # Ruta directa en HPC
    base_directory = f'{esquema_name}'
    folder_name = os.path.join(base_directory, f'ejecucion_cnn_iter_{iter}_cnn_{cnn_count}')

    os.makedirs(folder_name, exist_ok=True)
    log_file_path = os.path.join(folder_name, 'training_log.txt')
    model_save_path = os.path.join(folder_name, 'best_model.pth')

    # Definir el dispositivo: GPU si está disponible, sino CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    class CustomCNN(nn.Module):
        def __init__(self):
            super(CustomCNN, self).__init__()
            layers = []
            in_channels = 3
            current_filters = base_filter_value
            for _ in range(num_conv_layers):
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

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_path = '../dataset_procesado_img_cv/train'
    valid_path = '../dataset_procesado_img_cv/valid'
    test_path = '../dataset_procesado_img_cv/test'
    train_data = datasets.ImageFolder(train_path, transform=transform)
    valid_data = datasets.ImageFolder(valid_path, transform=transform)
    test_data = datasets.ImageFolder(test_path, transform=transform)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    model = CustomCNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_valid_accuracy = 0
    epochs_without_improvement = 0
    valid_accuracy_history = []
    train_loss_history, valid_loss_history = [], []

    with open(log_file_path, 'w') as log_file:
        log_file.write("Training Log\n=============")
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
            avg_train_loss = total_train_loss / len(train_loader.dataset)
            train_loss_history.append(avg_train_loss)

            model.eval()
            total_valid_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    total_valid_loss += loss.item() * data.size(0)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += data.size(0)
            avg_valid_loss = total_valid_loss / total
            valid_accuracy = 100. * correct / total
            valid_loss_history.append(avg_valid_loss)
            valid_accuracy_history.append(valid_accuracy)

            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                torch.save(model.state_dict(), model_save_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            log_file.write(f"\nEpoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%")

            if epochs_without_improvement >= early_stopping_patience:
                break

    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    y_true_all, y_pred_all, y_score_all = [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            probs = torch.exp(output)
            pred = output.argmax(dim=1, keepdim=True)
            y_true_all.extend(target.cpu().numpy())
            y_pred_all.extend(pred.cpu().numpy())
            y_score_all.extend(probs.cpu().numpy())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_score_all = np.array(y_score_all)

    conf_matrix = confusion_matrix(y_true_all, y_pred_all)
    recall = recall_score(y_true_all, y_pred_all, average='macro')
    auc_score = roc_auc_score(y_true_all, y_score_all[:, 1])
    f1 = f1_score(y_true_all, y_pred_all, average='macro')
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)

    end_time = time.time()
    total_execution_time = end_time - start_time
    final_avg_loss = sum(valid_loss_history) / len(valid_loss_history)
    final_avg_accuracy = sum(valid_accuracy_history) / len(valid_accuracy_history)

    # Guardar gráficos por cada CNN entrenada
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(valid_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_name, 'loss_per_epoch.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(valid_accuracy_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_name, 'accuracy_per_epoch.png'))
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(folder_name, 'confusion_matrix.png'))
    plt.close()

    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true_all, y_score_all[:, 1])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_name, 'roc_curve.png'))
    plt.close()

    metrics_names = ['Recall', 'Specificity', 'F1 Score', 'AUC']
    metrics_values = [recall, specificity, f1, auc_score]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics_names, metrics_values, color=['orange', 'cyan', 'magenta', 'brown'])
    plt.ylim(0, 1)
    plt.title('Resumen de Métricas Finales')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(folder_name, 'metric_summary_bar.png'))
    plt.close()

    hours, rem = divmod(total_execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"\n\nFinal Results\nFinal Average Validation Loss: {final_avg_loss:.4f}\n")
        log_file.write(f"Final Average Validation Accuracy: {final_avg_accuracy:.2f}%\n")
        log_file.write(f"Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1 Score: {f1:.4f}, AUC: {auc_score:.4f}\n")
        log_file.write(f"Total Execution Time: {time_str} (hh:mm:ss)\n")
        log_file.write(f"\nConfusion Matrix:\n{conf_matrix}\n")

    return final_avg_loss, final_avg_accuracy, recall, specificity, f1, auc_score, total_execution_time, model, y_true_all, y_pred_all, y_score_all, cnn_count

#************************************************************************************************************
#**********************************  G R E Y   W O L F   O P T I M A Z E R  *********************************
def ejecutarGWO(wolves,nwolves, maxIter):

    # Initialize variables to track history and best values
    history = []
    best_wolf = None
    best_accuracy = 0
    best_loss = float('inf')  # Initialize with a high value to minimize loss
    best_time = 0
    best_recall = 0
    best_specificity = 0
    best_f1 = 0
    best_auc = 0
    best_trained_model = None  # Guardar el mejor modelo de todas las iteraciones

    #best_model_save_path = '/content/drive/MyDrive/Colab Notebooks/{esquema_name}/best_model.pth'    *****comentado para HPC****
    best_model_save_path = f'{esquema_name}/History/best_model.pth'
   
    history_file_path = os.path.join(f'{esquema_name}/History/History_GWO.txt')
    # Asegurar que el directorio de historial existe, de lo contrario se crea
    os.makedirs(os.path.dirname(history_file_path), exist_ok=True)

    # Ensure the history file exists and is empty
    with open(history_file_path, 'w') as f:
        f.write("*************************")
        f.write("\nParametros de la ejecucion")
        f.write(f"\nNumero de lobos: {nwolves}")
        f.write(f"\nNumero de iteraciones: {maxIter}")
        f.write("\n*************************\n")
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
    fitness_scores = []

   

    #calcular antes de los movimientos y actualizar lobos
    cnn_count = 1  # Contador para las
    for wolf in wolves:

        print("\nLobo actual", wolf)

        final_average_valid_loss, final_average_valid_accuracy, recall, specificity, f1, auc_score, total_execution_time,trained_model = train_cnn_model(
            num_conv_layers=wolf['num_layers'],
            base_filter_value=wolf['num_filters'],
            use_batch_norm=wolf['batch_norm'],
            lr=wolf['lr'],
            batch_size=wolf['batch_size'],
            epochs=wolf['epochs'],
            iter = 0 ,            
            cnn_count=cnn_count
        )
        cnn_count = cnn_count + 1

        fitness_scores.append(final_average_valid_accuracy)

        print("\nAverage accuracy", final_average_valid_accuracy)
        print("\nAverage Loss", final_average_valid_loss)
        print("\nTiempo total en segundos ", total_execution_time)

        # Guardar el mejor valor
        if final_average_valid_accuracy is not None and final_average_valid_accuracy > best_accuracy:
            best_accuracy = final_average_valid_accuracy
            best_loss = final_average_valid_loss
            best_time = total_execution_time
            best_wolf = wolf.copy()
            best_f1=f1
            best_recall=recall
            best_specificity=specificity
            best_auc=auc_score
            best_trained_model=trained_model

    with open(history_file_path, 'a') as f:
        f.write("**************************************************\n")
        f.write(f"Mejor solucion antes de actualziar los parametros\n")
        f.write("**************************************************\n")
        f.write(f"Best wolf: {best_wolf}\n")
        f.write(f"Best Average Accuracy: {best_accuracy:.2f}%\n")
        f.write(f"Best Average Loss: {best_loss:.4f}\n")
        f.write(f"Best Recall: {best_recall:.4f}\n")
        f.write(f"Best Specificity: {best_specificity:.4f}\n")
        f.write(f"Best F1 Score: {best_f1:.4f}\n")
        f.write(f"Best AUC: {best_auc:.4f}\n")
        f.write(f"Best Execution Time: {best_time:.2f} seconds\n")
        f.write("\n")

    # Initialize counters for the CNN executions within this iteration
    print("********************************************************************************************************************")
    print("*********************************** SE COMIENZAN A ACTUALIZAR PARAMETROS CNN ***************************************")
    print("********************************************************************************************************************")
    print()

    for t in range(maxIter):
        print("********************************************************************************************************************")
        print("************************************************  ITERACION ", t+1, " **********************************************")
        print("********************************************************************************************************************")
        print()
        print(wolves)
        best_wolf_iteration = None
        best_accuracy_iteration = 0
        best_loss_iteration = 0
        best_time_iteration = 0
        best_f1_iteration = 0
        best_recall_iteration = 0
        best_specificity_iteration = 0
        best_auc_iteration = 0

        a = 2 - t * ((2) / maxIter)  # Disminución de a linealmente de 2 a 0

        # Ordenar lobos según fitness
        sorted_wolves = sorted(zip(fitness_scores, wolves), key=lambda x: x[0])
        alpha, beta, delta = sorted_wolves[0][1], sorted_wolves[1][1], sorted_wolves[2][1]


        # Actualizar posiciones de los lobos restantes
        for i, wolf in enumerate(wolves):
            for param in wolf:

                # Ignorar la actualización de Batch Norm (parámetro discreto)
                if param == 'batch_norm':
                    continue  # No actualizar el parámetro batch_norm (discreto)

                if isinstance(wolf[param], (int, float)):
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[param] - wolf[param])
                    X1 = alpha[param] - A1 * D_alpha

                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[param] - wolf[param])
                    X2 = beta[param] - A2 * D_beta

                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[param] - wolf[param])
                    X3 = delta[param] - A3 * D_delta

                    # Actualización de la posición
                    wolf[param] = (X1 + X2 + X3) / 3

                    # Truncar o redondear los parámetros que deben ser enteros
                    if param in ['num_layers', 'num_filters', 'batch_size', 'epochs']:
                        wolf[param] = int(round(wolf[param]))  # Convertir a entero redondeando
                        #que ningun parametro sea menor a 1
                        if wolf[param] < 1:
                            wolf[param] = 1
                    # Asegurar que el valor de lr esté en el rango permitido
                    if param == 'lr':
                        wolf[param] = max(0.0001, min(wolf[param], 0.01))  # Mantener lr dentro de [0.0001, 0.01]
                    
                    if param == 'epochs':
                        wolf[param] = max(20, min(wolf[param], 40))  
                    
                    if param == 'num_filters':
                        wolf[param] = max(32, min(wolf[param], 128))

                    if param == 'num_layers':
                        wolf[param] = max(2,min(wolf[param],5))

                    if param == 'batch_size':
                        wolf[param] = max(16,min(wolf[param],64))

        #calcular fitnees despues de actualizar lobos
        k = 0
        cnn_count = 1
        for wolf in wolves:
            print("*************************************************************************************************************")
            print("************************************************  LOBO ",k+1,"*************************************************")
            print("*************************************************************************************************************")
            print("\nLobo actual", wolf)
            k = k+1
            final_average_valid_loss, final_average_valid_accuracy, recall, specificity, f1, auc_score, total_execution_time,trained_model = train_cnn_model(
                num_conv_layers=wolf['num_layers'],
                base_filter_value=wolf['num_filters'],
                use_batch_norm=wolf['batch_norm'],
                lr=wolf['lr'],
                batch_size=wolf['batch_size'],
                epochs=wolf['epochs'],
                iter = t + 1,            
                cnn_count=cnn_count 
            )
            cnn_count = cnn_count + 1
             # Verificar si el entrenamiento fue válido
            if final_average_valid_accuracy is None:
                print(f"El lobo {wolf} fue descartado por dimensiones no válidas.")
                fitness_scores.append(0)  # Asignar un fitness score neutro
                continue  # Omitir este lobo y continuar con el siguiente
            fitness_scores.append(final_average_valid_accuracy)

            print("\nAverage accuracy", final_average_valid_accuracy)
            print("\nAverage Loss", final_average_valid_loss)
            print("\nTiempo total en segundos ", total_execution_time)

            
            if final_average_valid_accuracy is not None and final_average_valid_accuracy > best_accuracy_iteration:

                best_accuracy_iteration = final_average_valid_accuracy
                best_loss_iteration = final_average_valid_loss
                best_time_iteration = total_execution_time
                best_wolf_iteration = wolf
                best_f1_iteration=f1
                best_recall_iteration=recall
                best_specificity_iteration=specificity
                best_auc_iteration=auc_score
            # actualizar el mejor valor
            
            if final_average_valid_accuracy is not None and final_average_valid_accuracy > best_accuracy:
                print("SE CAMBIO EL MEJOR LOBO")
                best_accuracy = final_average_valid_accuracy
                best_loss = final_average_valid_loss
                best_time = total_execution_time
                best_wolf =  wolf.copy()
                best_recall=recall
                best_specificity=specificity
                best_auc=auc_score
                best_trained_model=trained_model

        # Guardar historial
        print("SE GUARDA ITERACION")
        with open(history_file_path, 'a') as f:
            f.write(f"\nIteration {t + 1}:\n")
            f.write(f"Best wolf: {best_wolf_iteration}\n")
            f.write(f"Best Average Accuracy: {best_accuracy_iteration:.2f}%\n")
            f.write(f"Best Average Loss: {best_loss_iteration:.4f}\n")
            f.write(f"Best Recall: {best_recall_iteration:.4f}\n")
            f.write(f"Best Specificity: {best_specificity_iteration:.4f}\n")
            f.write(f"Best F1 Score: {best_f1_iteration:.4f}\n")
            f.write(f"Best AUC: {best_auc_iteration:.4f}\n")
            f.write(f"Best Execution Time: {best_time_iteration:.2f} seconds\n")
            f.write("\n")

        best_accuracy_per_iteration.append(best_accuracy_iteration)
        best_loss_per_iteration.append(best_loss_iteration)
        best_execution_time_per_iteration.append(best_time_iteration)
        best_recall_per_iteration.append(best_recall_iteration)
        best_specificity_per_iteration.append(best_specificity_iteration)
        best_f1_per_iteration.append(best_f1_iteration)
        best_auc_per_iteration.append(best_auc_iteration)

     #guardar el mejor de todos
    with open(history_file_path, 'a') as f:
        f.write("******************\n")
        f.write(f"Mejor solucion\n")
        f.write("******************\n")
        f.write(f"Best wolf: {best_wolf}\n")
        f.write(f"Best Average Accuracy: {best_accuracy:.2f}%\n")
        f.write(f"Best Average Loss: {best_loss:.4f}\n")
        f.write(f"Best Recall: {best_recall:.4f}\n")
        f.write(f"Best Specificity: {best_specificity:.4f}\n")
        f.write(f"Best F1 Score: {best_f1:.4f}\n")
        f.write(f"Best AUC: {best_auc:.4f}\n")
        f.write(f"Best Execution Time: {best_time:.2f} seconds\n")
        f.write("\n")


    # Guardar el mejor modelo en la carpeta ra�z
    torch.save(best_trained_model.state_dict(), best_model_save_path)
    print(f"The best model of all iterations has been saved in: {best_model_save_path}")

    # Check if all metric lists have the correct length before plotting
    if len(best_recall_per_iteration) == maxIter:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, maxIter + 1), best_recall_per_iteration, marker='o', linestyle='-', color='orange')
        plt.title('Recall Evolution per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Recall')
        plt.grid(True)
        plt.savefig(os.path.join(f'{esquema_name}/History', 'recall_evolution_per_iteration.png'))
        #plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'recall_evolution_per_iteration.png'))   **** Comentado para HPC ****
        plt.close()

    # Repeat similar checks for other metrics
    if len(best_specificity_per_iteration) == maxIter:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, maxIter + 1), best_specificity_per_iteration, marker='o', linestyle='-', color='cyan')
        plt.title('Specificity Evolution per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Specificity')
        plt.grid(True)
        plt.savefig(os.path.join(f'{esquema_name}/History', 'specificity_evolution_per_iteration.png'))
        #plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'specificity_evolution_per_iteration.png'))
        plt.close()

    if len(best_f1_per_iteration) == maxIter:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, maxIter + 1), best_f1_per_iteration, marker='o', linestyle='-', color='magenta')
        plt.title('F1 Score Evolution per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('F1 Score')
        plt.grid(True)
        plt.savefig(os.path.join(f'{esquema_name}/History', 'f1_score_evolution_per_iteration.png'))
        #plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'f1_score_evolution_per_iteration.png'))
        plt.close()

    if len(best_auc_per_iteration) == maxIter:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, maxIter + 1), best_auc_per_iteration, marker='o', linestyle='-', color='brown')
        plt.title('AUC Evolution per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('AUC')
        plt.grid(True)
        plt.savefig(os.path.join(f'{esquema_name}/History', 'auc_evolution_per_iteration.png'))
        #plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'auc_evolution_per_iteration.png'))
        plt.close()

    return best_wolf, best_accuracy, best_loss, best_time, best_recall, best_specificity, best_f1, best_auc, best_accuracy_per_iteration, best_loss_per_iteration, best_execution_time_per_iteration,pd.DataFrame(history, columns=['Iteration', 'Nest Index', 'Current Nest', 'F(x) Current Value',
                                                    'F(x) New Value', 'New Nest', 'Status', 'Accuracy',
                                                    'Loss', 'Recall', 'Specificity', 'F1 Score', 'AUC', 'Execution Time'])
   

#******************************************************************** I N I C I O     A L G O R I T M O ***********************************************************************
#*******************************************************************************************************************************************************************************

# Inicializar población de lobos (soluciones aleatorias)
def initialize_population(param_ranges):
    wolf = {}
    for param, options in param_ranges.items():
        if isinstance(options, list):  # Si las opciones son una lista
            wolf[param] = random.choice(options)  # Selecciona un valor aleatorio de la lista
        elif isinstance(options, tuple):  # Para rangos num�ricos, como 'lr' y epochs
            wolf[param] = round(random.uniform(options[0], options[1]), 4)  # Genera un valor flotante aleatorio para lr, redondeado en 4 decimales
            if param == "epochs":
                wolf[param] = int(wolf[param])  # Asegurar que epochs sea un entero
    return wolf



# Inicialización de la población de lobos
wolves = [initialize_population(param_ranges) for _ in range(nwolves)]

print("poblacion de lobos")
for wolf in wolves:
    print(wolf)


# ejecutar GWO
start_time_gwo = time.time()  
best_wolf, best_accuracy, best_loss, best_time, best_recall, best_specificity, best_f1, best_auc , best_accuracy_per_iteration, best_loss_per_iteration, best_execution_time_per_iteration,history = ejecutarGWO(wolves,nwolves,maxIter)
end_time_gwo = time.time()

# Mostrar el historial de iteraciones y la mejor solucion encontrada
print("\nHistory de iteraciones:")
print(history.to_string(index=False))
print("\nBest Solution:")
print(f"wolf: {best_wolf}, Value de F(x): {best_accuracy}")
print(f"Best Average Accuracy: {best_accuracy:.2f}%")
print(f"Best Average Loss: {best_loss:.4f}")
print(f"Best Recall: {best_recall:.4f}")
print(f"Best Specificity: {best_specificity:.4f}")
print(f"Best F1 Score: {best_f1:.4f}")
print(f"Best AUC: {best_auc:.4f}")
print(f"Best Execution Time: {best_time:.2f} seconds")

# Tiempo total gwo
total_time_gwo = end_time_gwo - start_time_gwo

# Convert tiempo total de ejecucion en horas, minutos, y segundos
hours, rem = divmod(total_time_gwo, 3600)
minutes, seconds = divmod(rem, 60)
time_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
print('Total Total de ejecucion', time_formatted, '(hh:mm:ss)\n')

# Guardar el tiempo total de GWO en el archivo de historial
history_file_path = os.path.join(f'{esquema_name}/History/History_GWO.txt')
with open(history_file_path, 'a') as f:
    f.write("*************************\n")
    f.write(f"Tiempo total de ejecucion de GWO: {time_formatted} (hh:mm:ss)\n")
    f.write("*************************\n")

# Evoluci�n de la funci�n objetivo
# Guardar el gr�fico de la funci�n objetivo
plt.figure(figsize=(10, 6))
plt.plot(range(1, maxIter + 1), best_accuracy_per_iteration, marker='o', linestyle='-', color='blue')
plt.title('Best Value of the Target Function per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Best Value of the Target Function')
plt.grid(True)
plt.savefig(os.path.join(f'{esquema_name}/History', 'objective_function_evolution.png'))
#plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'objective_function_evolution.png'))  #**No para HPC ***
plt.close()

# Guardar el gr�fico de la mejor Accuracy por iteraci�n
plt.figure(figsize=(10, 6))
plt.plot(range(1, maxIter + 1), best_accuracy_per_iteration, marker='o', linestyle='-', color='green')
plt.title('Best Accuracy per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Best Accuracy (%)')
plt.grid(True)
plt.savefig(os.path.join(f'{esquema_name}/History', 'best_precision_per_iteration.png'))
#plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'best_precision_per_iteration.png'))  #**No para HPC ***
plt.close()

# Guardar el gr�fico de la mejor p�rdida por iteraci�n
plt.figure(figsize=(10, 6))
plt.plot(range(1, maxIter + 1), best_loss_per_iteration, marker='o', linestyle='-', color='red')
plt.title('Best Lost per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Best Lost')
plt.grid(True)
plt.savefig(os.path.join(f'{esquema_name}/History', 'best_loss_per_iteration.png'))
#plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'best_loss_per_iteration.png')) #**No para HPC ***
plt.close()

# Guardar el gr�fico del mejor tiempo de ejecuci�n por iteraci�n
plt.figure(figsize=(10, 6))
plt.plot(range(1, maxIter + 1), best_execution_time_per_iteration, marker='o', linestyle='-', color='purple')
plt.title('Best Execution Time per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Best Time (segundos)')
plt.grid(True)
plt.savefig(os.path.join(f'{esquema_name}/History', 'best_time_per_iteration.png'))
#plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'best_time_per_iteration.png'))    #**No para HPC ***
plt.close()

# Generate final graph of Recall, Specificity, F1, and AUC metrics with the best values
best_metric_values = [best_recall, best_specificity, best_f1, best_auc]
metric_labels = ['Recall', 'Specificity', 'F1 Score', 'AUC']
plt.figure(figsize=(10, 6))

# Se define paleta de colores para graficos de barra
pastel_colors = ['#aec6cf', '#ffb3ba', '#baffc9', '#ffdfba']  # Soft pastel shades of blue, pink, green, and orange
bars=plt.bar(metric_labels, best_metric_values, color=pastel_colors)
#bars=plt.bar(metric_labels, best_metric_values, color=['orange', 'cyan', 'magenta', 'brown'])
plt.title('Best metrics obtained at the end of execution')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom', fontsize=10, color='black')

plt.savefig(os.path.join(f'{esquema_name}/History', 'best_final_metrics.png'))
#plt.savefig(os.path.join('/content/drive/MyDrive/Colab Notebooks/{esquema_name}/History', 'best_final_metrics.png')) #**No para HPC ***
#plt.show()  #**No para HPC ***
plt.close()