import torch
import torch.nn as nn
from typing import Optional
from timm.layers import Mlp
from functools import partial



class ResNetBlock(nn.Module):
    """
    A ResNet block with adaptive layer norm zero (AdaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: Optional[float] = 4.0,
        **block_kwargs: dict,
    ):
        """
        Args:
            hidden_size: Number of features in the hidden layer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of hidden layer size in the MLP. None to skip the MLP.
            block_kwargs: Additional arguments to pass to the Attention block.
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp1 = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=partial(nn.GELU, approximate="tanh"),
        )
        self.use_mlp = mlp_ratio is not None
        if self.use_mlp:
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.mlp2 = Mlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                act_layer=partial(nn.GELU, approximate="tanh"),
            )
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer linear layers:
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.mlp1.apply(_basic_init)
        if self.use_mlp:
            self.mlp2.apply(_basic_init)

    def forward(
            self, 
            x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward pass of the ResNet block.
        In original implementation, conditioning is uniform across all tokens in the sequence. Here, we extend it to support token-wise conditioning (e.g. noise level can be different for each token).
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        """
        
        x = self.norm1(x)
        x = x + self.mlp1(x)
        if self.use_mlp:
            x = self.norm2(x)
            x = x + self.mlp2(x)
        return x
    

class ResNet(nn.Module):
    """
    ResNet that stacks n ResNetBlocks
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_blocks: int,
        mlp_ratio: float = 4.0,
    ):
        """
        Args:
            hidden_size: Hidden dimension for ResNet blocks
            num_blocks: Number of ResNet blocks to stack
            num_classes: Number of output classes
            mlp_ratio: MLP expansion ratio in ResNet blocks
            drop_rate: Dropout rate
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        
        # Stack of ResNet blocks
        self.blocks = nn.ModuleList([
            ResNetBlock(
                hidden_size=hidden_size,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_blocks)
        ])

        # self.output_block = nn.Linear(
        #     in_features=hidden_size,
        #     out_features=hidden_size,  # Output size can be adjusted as needed
        #     bias=True)
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        for block in self.blocks:
            block.initialize_weights()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, input_dim) where N is sequence length
        """       
        # Pass through ResNet blocks
        for block in self.blocks:
            x = block(x)
        
        # Final output block
        # x = self.output_block(x)
        
        return x