import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Configurações iguais ao CIFAR (32x32) ou LeNet padrão
# Se quiser mais detalhe, mude para 64x64
IMG_SIZE = 32 
BATCH_SIZE = 32
DATASET_PATH = "/home/nicolas/Documentos/topicos_sistemas_distribuidos/trabalho/DATASETS/Pokedex_v14"

def get_pokemon_dataloader():
    # 1. Transformações: Redimensionar para ficar "leve" igual CIFAR
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Força 32x32
        transforms.ToTensor(),
        # Normalização aproximada (pode ajustar depois calculando a média real do dataset)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Carregar igual ao CIFAR
    full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

    print(f"Dataset Carregado: {len(full_dataset)} imagens em {len(full_dataset.classes)} classes.")
    
    # Exemplo de como dividir treino/teste (80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, len(full_dataset.classes)

# Teste rápido
if __name__ == "__main__":
    train_loader, test_loader, num_classes = get_pokemon_dataloader()
    
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}") # Deve ser [32, 3, 32, 32]
    print(f"Classes totais: {num_classes}") # Deve ser 150