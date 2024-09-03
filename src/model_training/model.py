import torch
import torchvision.models as models

def create_model():
    model = models.resnet50(pretrained=True)
    
    # Unfreeze the last block only
    layers = list(model.children())
    
    # Print the names of layers to identify which ones to unfreeze
    # for i, layer in enumerate(layers):
    #     print(f"Layer {i}: {layer.__class__.__name__}")
    
    # Assuming layers[6] is 'layer1', layers[7] is 'layer2', layers[8] is 'layer3', and layers[9] is 'layer4'
    block_names = ['layer1', 'layer2', 'layer3', 'layer4']
    block_layers = layers[6:]  # 'layers[6]' onwards typically corresponds to the blocks
    
    # Unfreeze the last block only
    last_block = block_layers[-2]  # This should correspond to 'layer4'
    for param in last_block.parameters():
        param.requires_grad = True
    # print(f"Unfrozen block {block_names[-1]}")  # Print the name of the unfrozen block

    # Update fully connected layers
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
