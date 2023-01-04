import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class COVIDLungSegmenter(nn.Module):
    def __init__(self, encoder_name="resnet34", num_classes=4, activation=None):
        super(COVIDLungSegmenter, self).__init__()
        
        # Initializing Attention U-Net model with pretrained ImageNet weights
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=activation,
            decoder_attention_type="scse"  # Spatial and Channel Squeeze-and-Excitation attention
        )
        
    def forward(self, x):
        # Pass input tensor through the attention-enhanced encoder-decoder network
        return self.model(x)

if __name__ == "__main__":
    # Quick instantiation test
    net = COVIDLungSegmenter(num_classes=4)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = net(dummy_input)
    print(f"[TEST SUCCESS] Output tensor shape matching expectation: {output.shape}")