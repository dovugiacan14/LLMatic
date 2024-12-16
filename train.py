import os
import json
import importlib
import torch
import torch.utils
import torch.utils.data
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CIFAR100Trainer:
    def __init__(self, net_class, batch_size=64, learning_rate=0.001, momentum=0.9, num_epochs=10):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_epochs = num_epochs

        # prepare dataset
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.trainloader, self.testloader = self._load_data()
        self.net = net_class().to(device)
        self.criteration = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=self.learning_rate, momentum=self.momentum
        )

    def _load_data(self):
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False
        )

        return trainloader, testloader

    def train(self):
        try: 
            for epoch in range(self.num_epochs):
                self.net.train()
                running_loss = 0.0

                for inputs, labels in self.trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    self.optimizer.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.criteration(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                avg_loss = running_loss / len(self.trainloader)
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
            return True
        except Exception as e: 
            return False  

    def _calculate_accuracy(self, loader):
        self.net.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def evaluate(self):
        train_accuracy = self._calculate_accuracy(self.trainloader)
        test_accuracy = self._calculate_accuracy(self.testloader)
        return train_accuracy, test_accuracy


def get_modules_from_folder(folder):
    modules = []
    for file in os.listdir(folder):
        if file.endswith(".py") and file != "__init__.py":
            module_name = file[:-3]
            modules.append(module_name)
    return modules

if __name__ == "__main__":
    
    folder_path = "database_50/kaggle/working/database/"
    network_modules = get_modules_from_folder(folder_path)

    import sys
    sys.path.append(folder_path)

    results = []
    
    for module_name in network_modules:
        module = importlib.import_module(module_name)
        Net = getattr(module, "Net")
        trainer = CIFAR100Trainer(
            Net, batch_size=64, learning_rate=0.001, momentum=0.9, num_epochs=50
        )
        print(f"Training {module_name}...")
        trainer.train()
        
        train_acc, test_acc = trainer.evaluate()
        results.append({
        module_name: {
            "train_acc": f"{train_acc:.2f}%",
            "test_acc": f"{test_acc:.2f}%"
            }
        })
    print("Training results:", results)    

    # Save the results to a JSON file
    output_file = "result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results have been saved to {output_file}")