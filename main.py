import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from skimage import color, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. CLASSE DE FILTRAGEM (Orientada a Objetos)
# ==========================================
class FiltroNLM(object):
    def __init__(self, h_factor=0.0):
        """
        h_factor: Controla a agressividade do filtro.
        0.0 = Imagem Original (sem filtro)
        0.8 = Tratamento Leve
        1.2 = Tratamento Agressivo
        """
        self.h_factor = h_factor

    def __call__(self, img):
        # Se o fator for 0, pula a filtragem para economizar tempo
        if self.h_factor == 0.0:
            return img
            
        # Converte PIL Image para Numpy Array
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            img_np = color.rgb2gray(img_np)
            
        img_float = img_as_float(img_np)
        
        # Estima o ruído e aplica o NLM
        sigma_est = np.mean(estimate_sigma(img_float, channel_axis=None))
        img_filtrada = denoise_nl_means(img_float, h=self.h_factor * sigma_est, 
                                        fast_mode=True, patch_size=5, 
                                        patch_distance=7, channel_axis=None)
        
        # Retorna a imagem de volta para o formato PIL esperado pelo PyTorch
        return Image.fromarray((img_filtrada * 255).astype(np.uint8)).convert('RGB')

# ==========================================
# 2. CONFIGURAÇÃO DE HIPERPARÂMETROS E DADOS
# ==========================================
FATOR_RUIDO = 0.8 

print(f"Iniciando pipeline com h_factor = {FATOR_RUIDO} e Data Augmentation")

# Transformações de Treino (COM Augmentation)
transformacoes_treino = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    FiltroNLM(h_factor=FATOR_RUIDO),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Transformações de Validação (SEM Augmentation - Mundo Real)
transformacoes_val = transforms.Compose([
    transforms.Resize((224, 224)),
    FiltroNLM(h_factor=FATOR_RUIDO),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

diretorio_base = 'data'

datasets_img = {
    'train': datasets.ImageFolder(os.path.join(diretorio_base, 'train'), transformacoes_treino),
    'val': datasets.ImageFolder(os.path.join(diretorio_base, 'val'), transformacoes_val)
}

dataloaders = {x: torch.utils.data.DataLoader(datasets_img[x], batch_size=32, 
                                              shuffle=True, num_workers=0)
               for x in ['train', 'val']}

tamanhos_dataset = {x: len(datasets_img[x]) for x in ['train', 'val']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================
# 3. TRANSFER LEARNING (RESNET18)
# ==========================================
modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in modelo.parameters():
    param.requires_grad = False

num_ftrs = modelo.fc.in_features
modelo.fc = nn.Linear(num_ftrs, 2)
modelo = modelo.to(device)

criterio = nn.CrossEntropyLoss()
otimizador = optim.Adam(modelo.fc.parameters(), lr=0.001)

# ==========================================
# 4. LOOP DE TREINAMENTO (Com Histórico)
# ==========================================
def treinar_modelo(modelo, criterio, otimizador, epocas=3):
    desde = time.time()
    melhor_modelo_wts = copy.deepcopy(modelo.state_dict())
    melhor_acc = 0.0

    historico_loss_treino = []
    historico_loss_val = []
    historico_acc_treino = []
    historico_acc_val = []

    for epoca in range(epocas):
        print(f'Época {epoca+1}/{epocas}')
        print('-' * 10)

        for fase in ['train', 'val']:
            if fase == 'train':
                modelo.train()
            else:
                modelo.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[fase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                otimizador.zero_grad()

                with torch.set_grad_enabled(fase == 'train'):
                    outputs = modelo(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterio(outputs, labels)

                    if fase == 'train':
                        loss.backward()
                        otimizador.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / tamanhos_dataset[fase]
            epoch_acc = running_corrects.double() / tamanhos_dataset[fase]

            print(f'{fase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if fase == 'train':
                historico_loss_treino.append(epoch_loss)
                historico_acc_treino.append(epoch_acc.item())
            else:
                historico_loss_val.append(epoch_loss)
                historico_acc_val.append(epoch_acc.item())

            if fase == 'val' and epoch_acc > melhor_acc:
                melhor_acc = epoch_acc
                melhor_modelo_wts = copy.deepcopy(modelo.state_dict())

        print()

    tempo_decorrido = time.time() - desde
    print(f'Treinamento concluído em {tempo_decorrido // 60:.0f}m {tempo_decorrido % 60:.0f}s')
    print(f'Melhor Acurácia de Validação: {melhor_acc:4f}')

    modelo.load_state_dict(melhor_modelo_wts)
    return modelo, historico_loss_treino, historico_loss_val, historico_acc_treino, historico_acc_val

# ==========================================
# 5. FUNÇÕES PARA GERAR OS GRÁFICOS
# ==========================================
def plotar_resultados(hist_loss_t, hist_loss_v, hist_acc_t, hist_acc_v):
    epocas = range(1, len(hist_loss_t) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epocas, hist_loss_t, 'b-', marker='o', label='Treino Loss')
    ax1.plot(epocas, hist_loss_v, 'r-', marker='o', label='Validação Loss')
    ax1.set_title('Evolução do Erro (Loss)')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epocas, hist_acc_t, 'b-', marker='o', label='Treino Acc')
    ax2.plot(epocas, hist_acc_v, 'r-', marker='o', label='Validação Acc')
    ax2.set_title('Evolução da Acurácia')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Acurácia')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('curvas_aprendizagem.png')
    plt.show()

def gerar_matriz_confusao(modelo, dataloader_val):
    modelo.eval()
    todas_preds = []
    todas_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader_val:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = modelo(inputs)
            _, preds = torch.max(outputs, 1)
            
            todas_preds.extend(preds.cpu().numpy())
            todas_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(todas_labels, todas_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Pneumonia'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão - Validação')
    plt.savefig('matriz_confusao.png')
    plt.show()

# ==========================================
# 6. EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == '__main__':
    # Treina o modelo e guarda o histórico
    modelo_treinado, loss_t, loss_v, acc_t, acc_v = treinar_modelo(modelo, criterio, otimizador, epocas=3)

    # Gera e salva os gráficos
    print("\nGerando gráficos...")
    plotar_resultados(loss_t, loss_v, acc_t, acc_v)
    gerar_matriz_confusao(modelo_treinado, dataloaders['val'])
    print("Gráficos salvos como 'curvas_aprendizagem.png' e 'matriz_confusao.png' na pasta do projeto.")