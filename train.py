import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CustomCNN

def main():
    # Data loaders
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    # Model, Loss, Optimizer
    model = CustomCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    #20 epochs
    for epoch in range(20):  
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        epoch_train_accuracy = 100 * correct_train / total_train

        print(f'Epoch {epoch + 1} finished. Training Accuracy: {epoch_train_accuracy:.2f}%')
    # Save the model
    torch.save(model.state_dict(), "model.pth")
    print("Finished Training and saved model")

if __name__ == '__main__':
    main()