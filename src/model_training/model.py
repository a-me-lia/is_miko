import torch
import torchvision.models as models

def create_model():
    model = models.resnet50(pretrained=True)
    layers = list(model.children())

    block_names = ['layer1', 'layer2', 'layer3', 'layer4']
    block_layers = layers[6:]  

    last_block = block_layers[-2] # unfreeze the top two layers [resnet residual block]
    for param in last_block.parameters():
        param.requires_grad = True

    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    )
    
    return model

# Create the model and print the unfrozen block
model = create_model()
