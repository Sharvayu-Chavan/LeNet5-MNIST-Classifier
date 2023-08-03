#Basic Modules
import torch

#Datasets, Dataloaders, Preprocessing Modules
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

#Model Architecture Modules
import torch.nn as nn

#Data Transformation for LeNet Architecture
trans = transforms.Compose([
    # To resize image
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    # To normalize image
    transforms.Normalize((0.5,), (0.5,))
])

#Importing Datasets
training_data = datasets.MNIST(root='./data',train = True, download = True, transform=trans)
testing_data = datasets.MNIST(root='./data',train = False, download = True, transform=trans)

#Defining Dataloaders
batch_size = 64
train_dataloader = DataLoader(training_data,batch_size=batch_size)
test_dataloader = DataLoader(testing_data, batch_size=batch_size)
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#Define Device for model training and testing                                                           LOOK HERE AND FIX   
def device_used():
    if torch.cuda.is_available():
        print('gpu avaliable')
        return 'cuda'
    else:
        print('gpu not avaliable')
        return 'cpu'
device = device_used()
print(f"using {device} device")


#Define Model Architecture and Model
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(6, 16, kernel_size = 5, stride = 1, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)  
        )
    def forward(self, x):
        x = self.convolutional_layers(x)
        x = torch.flatten(x, 1)
        logits = self.fc_layers(x)
        return logits
model = LeNet().to(device)

#Define Loss Function and Optimization Algorithm
learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Define Training Loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        #Create prediction and loss from Forward propagation
        pred = model(X)
        loss = loss_fn(pred,y)
        
        #Backpropagation Portion
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader,model,loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#Define Training Iterations
epochs = 30
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

#save model
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

#loading model
model = LeNet().to(device)
model.load_state_dict(torch.load("model.pth"))
print("Loaded PyTorch Model State from model.pth")