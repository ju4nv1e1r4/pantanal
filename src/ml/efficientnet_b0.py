import torch
import torch.nn as nn
import timm

class DeepWetlandsModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=234, pretrained=True):
        super(DeepWetlandsModel, self).__init__()

        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=1
        )
        
    def forward(self, x):
        # x shape: [batch, 1, freq, time]
        logits = self.model(x)
        return logits

if __name__ == "__main__":
    # fast sanity test
    model = DeepWetlandsModel()
    dummy_input = torch.randn(8, 1, 128, 313)
    output = model(dummy_input)
    
    print(f"Model: {type(model.model).__name__}")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape (Logits): {output.shape}") # must be [8, 234]

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {params:,}")
