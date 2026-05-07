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
# Mantemos o Fator 0.8 que deu o melhor resultado
FATOR_RUIDO = 0.8 

print(f"Iniciando pipeline com h_factor = {FATOR_RUIDO} e Data Augmentation")

# 1. Transformações de Treino (COM Augmentation)
transformacoes_treino = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), # 50% de chance de espelhar horizontalmente
    transforms.RandomRotation(degrees=10),  # Rotaciona a imagem entre -10 e +10 graus
    FiltroNLM(h_factor=FATOR_RUIDO),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Transformações de Validação (SEM Augmentation - Mundo Real)
transformacoes_val = transforms.Compose([
    transforms.Resize((224, 224)),
    FiltroNLM(h_factor=FATOR_RUIDO),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

diretorio_base = 'data'

# Aplicando as transformações corretas para cada pasta
datasets_img = {
    'train': datasets.ImageFolder(os.path.join(diretorio_base, 'train'), transformacoes_treino),
    'val': datasets.ImageFolder(os.path.join(diretorio_base, 'val'), transformacoes_val)
}

# DataLoader cuida do batching (enviar de 32 em 32 imagens)
dataloaders = {x: torch.utils.data.DataLoader(datasets_img[x], batch_size=32, 
                                              shuffle=True, num_workers=0)
               for x in ['train', 'val']}

tamanhos_dataset = {x: len(datasets_img[x]) for x in ['train', 'val']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================
# 3. TRANSFER LEARNING (RESNET18)
# ==========================================
modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Congela os pesos das camadas convolucionais (já sabem extrair características)
for param in modelo.parameters():
    param.requires_grad = False

# Substitui a última camada (Totalmente Conectada) para as nossas 2 classes
num_ftrs = modelo.fc.in_features
modelo.fc = nn.Linear(num_ftrs, 2)
modelo = modelo.to(device)

criterio = nn.CrossEntropyLoss()
otimizador = optim.Adam(modelo.fc.parameters(), lr=0.001)

# ==========================================
# 4. LOOP DE TREINAMENTO
# ==========================================
def treinar_modelo(modelo, criterio, otimizador, epocas=3):
    desde = time.time()
    melhor_modelo_wts = copy.deepcopy(modelo.state_dict())
    melhor_acc = 0.0

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

            # Itera sobre os dados
            for inputs, labels in dataloaders[fase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                otimizador.zero_grad()

                # Forward
                with torch.set_grad_enabled(fase == 'train'):
                    outputs = modelo(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterio(outputs, labels)

                    # Backward + Otimização apenas no treino
                    if fase == 'train':
                        loss.backward()
                        otimizador.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / tamanhos_dataset[fase]
            epoch_acc = running_corrects.double() / tamanhos_dataset[fase]

            print(f'{fase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Salva o melhor modelo
            if fase == 'val' and epoch_acc > melhor_acc:
                melhor_acc = epoch_acc
                melhor_modelo_wts = copy.deepcopy(modelo.state_dict())

        print()

    tempo_decorrido = time.time() - desde
    print(f'Treinamento concluído em {tempo_decorrido // 60:.0f}m {tempo_decorrido % 60:.0f}s')
    print(f'Melhor Acurácia de Validação: {melhor_acc:4f}')

    modelo.load_state_dict(melhor_modelo_wts)
    return modelo

# Executa o treino (coloquei 3 épocas para rodar rápido)
modelo_treinado = treinar_modelo(modelo, criterio, otimizador, epocas=3)