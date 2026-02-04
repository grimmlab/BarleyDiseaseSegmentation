"""
UNet architecture with flexible encoder backbone for barley disease segmentation.
"""

import timm
import torch.nn as nn
from barley_disease_segmentation.config import *

__all__ = ['DecoderBlock', 'FlexibleUNet']


class DecoderBlock(nn.Module):
    """Decoder block for UNet architecture."""

    def __init__(self, in_channels, skip_channels, out_channels, dropout_rate=0.0):
        super().__init__()

        # Upsampling layer
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        # Convolution block after concatenation with skip connection
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            # Handle potential size mismatches
            if x.shape[2:] != skip.shape[2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class FlexibleUNet(nn.Module):
    """Flexible UNet segmentation model with configurable encoder backbone."""

    def __init__(self,
                 encoder_name: str = "resnet34",
                 num_classes: int = 1,
                 task_type: str = "binary",
                 bottleneck_dropout_rate: float = 0.0,
                 decoder_dropout_rate: float = 0.0,
                 input_size: int = 512):
        """
        Initialize a flexible UNet model with various encoders from timm.

        Args:
            encoder_name: Name of the encoder from timm library ( 'resnet34', 'efficientnet_b2', 'convnext_tiny')
            convnext_tiny.fb_in1k; 'efficientnet_b2.ra_in1k'; 'resnet34.a1_in1k'
            num_classes: Number of output classes
                - Binary: automatically uses 2 output channels (background + lesion)
                - Multiclass: uses specified num_classes channels
            task_type: "binary" or "multiclass"
            bottleneck_dropout_rate: Dropout rate for bottleneck
            decoder_dropout_rate: Dropout rate for decoder blocks
            input_size: Input image size (assumes square images)
        """
        super().__init__()

        self.task_type = task_type
        self.num_classes = num_classes
        self.input_size = input_size

        # Validate task type and set appropriate output channels
        if task_type == "binary":
            output_channels = 2  # Two channels for binary segmentation
        elif task_type == "multiclass":
            if num_classes < 2:
                raise ValueError("For multiclass task, num_classes must be ≥ 2")
            output_channels = num_classes  # Multiple channels for multiclass
        else:
            raise ValueError("task_type must be 'binary' or 'multiclass'")

        # Create encoder
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=True,
            features_only=True,
            out_indices=None
        )

        # Get encoder feature channels
        self.encoder_channels = self.encoder.feature_info.channels()
        print(f"Encoder {encoder_name} feature channels: {self.encoder_channels}")

        # Reverse for decoder (deepest to shallowest)
        decoder_channels = list(reversed(self.encoder_channels))
        self.enc_out_norm = nn.BatchNorm2d(decoder_channels[0])  #

        # Bridge: process the deepest feature map
        self.bridge = nn.Sequential(
            nn.Conv2d(decoder_channels[0], decoder_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[0], decoder_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.bottleneck_dropout = nn.Dropout2d(
            bottleneck_dropout_rate) if bottleneck_dropout_rate > 0 else nn.Identity()

        # Create decoders - one for each skip connection
        self.decoders = nn.ModuleList()

        for i in range(len(decoder_channels) - 1):
            self.decoders.append(DecoderBlock(
                in_channels=decoder_channels[i],
                skip_channels=decoder_channels[i + 1],
                out_channels=decoder_channels[i + 1],
                dropout_rate=decoder_dropout_rate
            ))

        # Final upsampling to get close to input size
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[-1], decoder_channels[-1],
                               kernel_size=2, stride=2),
            nn.Conv2d(decoder_channels[-1], 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final convolution to get to the appropriate number of output channels
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Input shape: [batch_size, 3, height, width]
        original_size = x.shape[2:]
        x = x.float()

        # Get encoder features
        features = self.encoder(x)
        features_rev = list(reversed(features))  # Reverse for decoder

        # Bridge
        deep_feat = features_rev[0]
        deep_feat = self.enc_out_norm(deep_feat)  # stabilize deepest encoder output
        x = self.bridge(deep_feat)
        x = self.bottleneck_dropout(x)

        # Decoder path with skip connections
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, features_rev[i + 1])

        # Final upsampling
        x = self.final_upsample(x)

        # Ensure output matches input spatial dimensions
        if x.shape[2:] != original_size:
            x = torch.nn.functional.interpolate(
                x, size=original_size, mode='bilinear', align_corners=False
            )

        # Final output:
        # - Binary: [batch_size, 1, height, width] (sigmoid probabilities)
        # - Multiclass: [batch_size, num_classes, height, width] (softmax probabilities)
        logits = self.final_conv(x)

        return logits