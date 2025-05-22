import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CustomCNN

def main():
    # Load test data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # Load model
    model = CustomCNN()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    main()