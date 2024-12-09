import torch
import torchvision
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# prepare dataset 
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
trainset = torchvision.datasets.CIFAR10(
    root= "./data", train= True, download= True, transform= transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size= 64, shuffle= True 
)

def is_trainable(net):
    try: 
        net.to(device)

        inputs, labels = next(iter(trainloader))
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        if outputs.shape[0] != labels.shape[0]:
            print(f"Output batch size mismatch: expected {labels.shape[0]}, got {outputs.shape[0]}")
            return False
        
        if outputs.shape[-1] != 10:
            print(f"Output shape mismatch: expected last dimension 10, got {outputs.shape[-1]}")
            return False
        return True 
    
    except Exception as e: 
        print(f"Error occurred when checking trainable: {e}")
        return False 