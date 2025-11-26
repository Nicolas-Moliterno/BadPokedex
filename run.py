import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split # Adicionado random_split
import matplotlib.pyplot as plt
import numpy as np
import time

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
DATASET_PATH = "/home/nicolas/Documentos/topicos_sistemas_distribuidos/trabalho/DATASETS/Pokedex_v14"
IMG_SIZE = 32      # Tamanho nativo da LeNet clássica
BATCH_SIZE = 32    # Batch para treino. Para IDLG (abaixo), será 1
ITERATIONS = 1000  # Quantos passos de otimização para reconstruir a imagem
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TRAIN_EPOCHS = 300 # Número de épocas para pré-treinar a LeNet

print(f"Usando dispositivo: {DEVICE}")

# ==========================================
# 2. DATALOADER
# ==========================================
# Esta função agora retorna loaders e o número total de classes.
# get_one_pokemon_image foi incorporado para simplificar.
def get_pokemon_dataloaders(batch_size_attack=1):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
    
    # Dividir em treino e teste (80/20) para o pré-treino do modelo
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # DataLoader para o ataque (uma imagem por vez, sem shuffle)
    attack_loader = DataLoader(full_dataset, batch_size=batch_size_attack, shuffle=True) # Usar o dataset completo para pegar qq imagem
    
    num_classes = len(full_dataset.classes)
    
    print(f"Dataset Carregado: {len(full_dataset)} imagens em {num_classes} classes.")
    
    return train_loader, test_loader, attack_loader, num_classes

# ==========================================
# 3. MODELO (LeNet corrigida para 32x32)
# ==========================================
class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Sigmoid nas convoluções é o "segredo" do paper DLG original
        # Isso evita gradientes mortos (Zero) da ReLU
        act = nn.Sigmoid()
        
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=2),
            act,
            nn.AvgPool2d(2), # Pooling suave é melhor que MaxPool para gradientes
            nn.Conv2d(12, 12, kernel_size=5, padding=2),
            act,
            nn.AvgPool2d(2),
        )
        
        # 12 canais * 8 * 8 = 768 (Entrada 32x32 -> 16x16 -> 8x8)
        self.fc = nn.Sequential(
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def weights_init(m):
    if hasattr(m, "weight") and m.weight.requires_grad:
        nn.init.xavier_uniform_(m.weight.data) # Inicialização melhor
    if hasattr(m, "bias") and m.bias is not None:
        m.bias.data.zero_() # Bias para zero

# ==========================================
# 4. FUNÇÃO DE TREINAMENTO (NOVA)
# ==========================================
def train_model(net, train_loader, criterion, num_epochs):
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    net.train() # Coloca o modelo em modo de treino
    print(f"\nIniciando treinamento da LeNet por {num_epochs} épocas para o ataque...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print a cada 100 mini-batches ou no final da época
            if i % 100 == 99 or i == len(train_loader) -1:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1:5d} loss: {running_loss / (i+1):.3f}")
        # print(f"Epoch {epoch + 1} Loss Média: {running_loss / len(train_loader):.3f}")
    print("Treinamento concluído.")

# ==========================================
# 5. O ATAQUE IDLG (MODIFICADO)
# ==========================================
def run_attack():
    # A. Preparar Dados e Dataloaders
    train_loader, _, attack_loader, num_classes = get_pokemon_dataloaders(batch_size_attack=1)
    
    # Pegar uma imagem aleatória para atacar
    gt_data, gt_label = next(iter(attack_loader))
    gt_data, gt_label = gt_data.to(DEVICE), gt_label.to(DEVICE)

    # B. Preparar Modelo
    net = LeNet(num_classes=num_classes).to(DEVICE)
    net.apply(weights_init)
    criterion = nn.CrossEntropyLoss()
    
    # --- NOVO: TREINAR O MODELO ANTES DO ATAQUE ---
    train_model(net, train_loader, criterion, num_epochs=NUM_TRAIN_EPOCHS)
    net.eval() # Mudar para modo de avaliação para o ataque (desliga dropout/batchnorm se houver)

    print(f"\nAlvo carregado. Classe Real (Índice): {gt_label.item()}")
    
    # C. Calcular o Gradiente Original
    pred = net(gt_data)
    y_loss = criterion(pred, gt_label)
    original_dy_dx = torch.autograd.grad(y_loss, net.parameters())
    original_dy_dx = [g.detach() for g in original_dy_dx]

    # D. Dados Dummy
    # Inicialização "cinza" ajuda na convergência
    dummy_data = torch.randn(gt_data.size()).to(DEVICE) * 0.1 # <-- Inicialização mais suave
    dummy_data.requires_grad_(True)
    
    # Inferência de label
    last_weight_grad = original_dy_dx[-2]
    label_pred = torch.argmin(torch.sum(last_weight_grad, dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
    
    print(f"Label Inferido pelo gradiente: {label_pred.item()} | Label Real: {gt_label.item()}")
    if label_pred.item() != gt_label.item():
        print("Aviso: A inferência de label errou, forçando label real para teste de reconstrução.")
        label_pred = gt_label

    # E. Otimizador para reconstrução
    optimizer = torch.optim.Adam([dummy_data], lr=0.1)

    # Adicione a Loss de Regularização (Total Variation - TV Loss) para remover ruído
    # Essa Loss penaliza pixels que mudam bruscamente de cor
    tv_loss = lambda x: torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
                        torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    
    alpha = 0.001 # Peso da Loss de Regularização (valor padrão)
    print(f"\nIniciando reconstrução via gradientes (ADAM, {ITERATIONS} iterações)...")
    history = []
    
    for iters in range(ITERATIONS):
        optimizer.zero_grad()
        
        # Clamp (segurança)
        with torch.no_grad():
             dummy_data.clamp_(-1, 1) 

        # 1. Calcular a Loss de Classificação no dummy
        dummy_pred = net(dummy_data) 
        dummy_loss = criterion(dummy_pred, label_pred)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        # 2. Calcular a diferença de Gradientes (MSE)
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()

        # 3. Adicionar a Loss de Regularização (TV Loss)
        total_loss = grad_diff + alpha * tv_loss(dummy_data) # <--- NOVO
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        if iters % 100 == 0:
            print(f"Iter {iters}: Loss do Gradiente = {grad_diff.item():.6f}")

        # Se a Loss for menor que um valor pequeno, pare
        if grad_diff.item() < 0.01:
            print(f"Convergência Rápida em {iters} iterações.")
            break

        # Atualiza a Loss final
        loss_val = grad_diff
        
        if iters % 50 == 0:
            current_loss = loss_val.item()
            print(f"Iter {iters}: Loss do Gradiente = {current_loss:.6f}")
            history.append(dummy_data.detach().cpu().clone())

    # ==========================================
    # 6. VISUALIZAÇÃO DOS RESULTADOS
    # ==========================================
    print("Plotando resultados...")
    
    def tensor_to_img(t):
        t = t.squeeze(0).detach().cpu().numpy()
        t = (t * 0.5) + 0.5
        t = np.clip(t, 0, 1)
        return np.transpose(t, (1, 2, 0))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(tensor_to_img(gt_data))
    axes[0].set_title("Original (Dataset)")
    
    axes[1].imshow(tensor_to_img(history[0])) 
    axes[1].set_title("Ruído Inicial")
    
    axes[2].imshow(tensor_to_img(dummy_data))
    axes[2].set_title(f"Reconstruído (Iter {ITERATIONS}) - Final Loss: {loss_val.item():.2f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_attack()