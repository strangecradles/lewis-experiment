"""
Model loading and feature extraction for Lewis superadditivity experiment.

Three frozen vision models with different representational biases:
- DINOv2 ViT-S/14: spatial structure, object boundaries, layout
- SigLIP ViT-B/16: semantic categories, language-aligned concepts  
- MAE ViT-B/16: low-level texture, color, material properties

All models return patch tokens (not just CLS). Frozen throughout training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import timm
import open_clip
from torchvision import transforms
import warnings


@dataclass
class ModelConfig:
    """Configuration for a single vision model."""
    name: str
    source: str  # 'timm' or 'open_clip'
    model_name: str
    patch_size: int
    embed_dim: int
    image_size: int = 224


class FeatureCache:
    """Pre-extracted features from frozen models, stored in CPU memory (float16)."""

    def __init__(
        self,
        cls_tokens: Dict[str, torch.Tensor],
        patch_tokens: Dict[str, torch.Tensor],
        image_id_to_idx: Dict[str, int],
    ):
        self.cls_tokens = cls_tokens      # {model: [N, D]}
        self.patch_tokens = patch_tokens  # {model: [N, P, D]}
        self.image_id_to_idx = image_id_to_idx

    def get_batch(
        self,
        model_names: List[str],
        indices: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Gather features for a batch of image indices and move to device as float32."""
        features: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in model_names:
            features[name] = {
                'cls_token': self.cls_tokens[name][indices].float(),
                'patch_tokens': self.patch_tokens[name][indices].float(),
            }
        return features
    

class ModelBank:
    """
    Manages three frozen vision models with different representational biases.
    
    Loads all models once at init and provides unified feature extraction interface.
    Target: GH200 with 141GB unified memory - keep all models loaded.
    """
    
    MODELS = {
        'dino': ModelConfig(
            name='dino',
            source='timm',
            model_name='vit_small_patch14_dinov2.lvd142m',
            patch_size=14,
            embed_dim=384,
            image_size=518
        ),
        'siglip': ModelConfig(
            name='siglip',
            source='open_clip',
            model_name='ViT-B-16-SigLIP',
            patch_size=16,
            embed_dim=768,
            image_size=224
        ),
        'mae': ModelConfig(
            name='mae',
            source='timm',
            model_name='vit_base_patch16_224',
            patch_size=16,
            embed_dim=768,
            image_size=224
        )
    }
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.models = {}
        self.preprocessors = {}
        
        print("Loading vision models...")
        self._load_models()
        print(f"Loaded {len(self.models)} models on {device}")
    
    def _load_models(self):
        """Load all three vision models and their preprocessors."""
        for model_name, config in self.MODELS.items():
            print(f"  Loading {config.name} ({config.model_name})...")
            
            if config.source == 'timm':
                model = self._load_timm_model(config)
                preprocessor = self._get_timm_preprocessor(config)
            elif config.source == 'open_clip':
                model = self._load_open_clip_model(config)
                preprocessor = self._get_open_clip_preprocessor(config)
            else:
                raise ValueError(f"Unknown model source: {config.source}")
            
            # Freeze model
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            self.models[model_name] = model.to(self.device)
            self.preprocessors[model_name] = preprocessor
    
    def _load_timm_model(self, config: ModelConfig) -> nn.Module:
        """Load a model from timm."""
        try:
            model = timm.create_model(
                config.model_name,
                pretrained=True,
                num_classes=0,  # Remove classification head
            )
            return model
        except Exception as e:
            # Fallback for different timm versions
            warnings.warn(f"Failed to load {config.model_name}, trying without suffix")
            base_name = config.model_name.split('.')[0]
            model = timm.create_model(
                base_name,
                pretrained=True,
                num_classes=0,
            )
            return model
    
    def _load_open_clip_model(self, config: ModelConfig) -> nn.Module:
        """Load a model from open_clip."""
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.model_name,
            pretrained='webli'
        )
        # Extract the timm VisionTransformer trunk from the open_clip wrapper
        return model.visual.trunk
    
    def _get_timm_preprocessor(self, config: ModelConfig) -> transforms.Compose:
        """Get preprocessing pipeline for timm models."""
        # Standard ImageNet preprocessing
        return transforms.Compose([
            transforms.Resize(
                (config.image_size, config.image_size), 
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
    
    def _get_open_clip_preprocessor(self, config: ModelConfig) -> transforms.Compose:
        """Get preprocessing pipeline for open_clip models."""
        return transforms.Compose([
            transforms.Resize(
                (config.image_size, config.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
    
    def _extract_features_timm(self, model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from timm model, returning (patch_tokens, cls_token)."""
        # Forward through patch embedding
        x = model.patch_embed(x)
        
        # Add cls token (some models like SigLIP set cls_token=None)
        has_cls = hasattr(model, 'cls_token') and model.cls_token is not None
        if has_cls:
            cls_token = model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        # Add positional embeddings
        if hasattr(model, 'pos_embed'):
            x = x + model.pos_embed
        
        # Apply dropout
        if hasattr(model, 'pos_drop'):
            x = model.pos_drop(x)
        
        # Forward through transformer blocks
        if hasattr(model, 'blocks'):
            for block in model.blocks:
                x = block(x)
        
        # Apply final norm
        if hasattr(model, 'norm'):
            x = model.norm(x)
        
        # Split cls and patch tokens
        if has_cls:
            cls_token = x[:, 0]  # [batch, embed_dim]
            patch_tokens = x[:, 1:]  # [batch, num_patches, embed_dim]
        else:
            # No cls token (e.g. SigLIP), use global average pooling
            patch_tokens = x
            cls_token = x.mean(dim=1)
        
        return patch_tokens, cls_token
    
    @torch.no_grad()
    def get_features(self, 
                    images: torch.Tensor, 
                    model_names: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract features from specified models.
        
        Args:
            images: Batch of images [batch_size, 3, H, W] (raw images, will be preprocessed)
            model_names: List of model names to extract features from
            
        Returns:
            Dict mapping model_name -> {'patch_tokens': Tensor, 'cls_token': Tensor}
            - patch_tokens: [batch_size, num_patches, embed_dim]
            - cls_token: [batch_size, embed_dim]
        """
        features = {}
        
        for model_name in model_names:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")
            
            model = self.models[model_name]
            preprocessor = self.preprocessors[model_name]
            config = self.MODELS[model_name]
            
            # Preprocess images (assuming they come in as PIL or tensor format)
            if isinstance(images, torch.Tensor) and images.max() > 1.0:
                # Convert from [0, 255] to [0, 1] if needed
                processed_images = images / 255.0
            else:
                processed_images = images
                
            # Apply model-specific preprocessing
            if processed_images.shape[1:] != (3, config.image_size, config.image_size):
                # Resize if needed
                processed_images = F.interpolate(
                    processed_images, 
                    size=(config.image_size, config.image_size), 
                    mode='bicubic', 
                    align_corners=False
                )
            
            # Normalize with model-specific mean/std
            if config.source == 'open_clip':
                mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(images.device)
                std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(images.device)
            else:  # timm (ImageNet)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            
            processed_images = (processed_images - mean) / std
            
            # Extract features (all models use timm-style ViT trunk)
            patch_tokens, cls_token = self._extract_features_timm(model, processed_images)
            
            features[model_name] = {
                'patch_tokens': patch_tokens,
                'cls_token': cls_token
            }
        
        return features
    
    @torch.no_grad()
    def precompute_features(
        self,
        image_lookup: Dict[str, 'Image.Image'],
        batch_size: int = 32,
    ) -> FeatureCache:
        """Pre-extract features from all models for all images.

        Processes images in batches through all 3 frozen models and stores
        results as float16 tensors in CPU memory.
        """
        from PIL import Image as PILImage

        image_ids = sorted(image_lookup.keys())
        image_id_to_idx = {img_id: idx for idx, img_id in enumerate(image_ids)}
        num_images = len(image_ids)
        model_names = list(self.MODELS.keys())

        print(f"Pre-extracting features for {num_images} images across {len(model_names)} models...")

        to_tensor = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

        cls_lists: Dict[str, List[torch.Tensor]] = {n: [] for n in model_names}
        patch_lists: Dict[str, List[torch.Tensor]] = {n: [] for n in model_names}

        for batch_start in range(0, num_images, batch_size):
            batch_end = min(batch_start + batch_size, num_images)
            batch_ids = image_ids[batch_start:batch_end]

            tensors = []
            for img_id in batch_ids:
                img = image_lookup[img_id]
                if not isinstance(img, PILImage.Image):
                    img = PILImage.new('RGB', (224, 224))
                tensors.append(to_tensor(img.convert('RGB')))

            batch = torch.stack(tensors).to(self.device)
            features = self.get_features(batch, model_names)

            for name in model_names:
                cls_lists[name].append(features[name]['cls_token'].half())
                patch_lists[name].append(features[name]['patch_tokens'].half())

            done = batch_end
            if done % (batch_size * 100) < batch_size or done == num_images:
                print(f"  {done}/{num_images} images processed")

        cls_tokens = {n: torch.cat(ts) for n, ts in cls_lists.items()}
        patch_tokens = {n: torch.cat(ts) for n, ts in patch_lists.items()}

        total_gb = sum(t.nbytes for t in cls_tokens.values()) + sum(t.nbytes for t in patch_tokens.values())
        total_gb /= 1e9
        for name in model_names:
            print(f"  {name}: cls {cls_tokens[name].shape}, patches {patch_tokens[name].shape}")
        print(f"  Total cache size: {total_gb:.1f} GB (float16, on GPU)")

        # Free frozen models from GPU — no longer needed
        print("  Freeing frozen models from GPU...")
        self.models.clear()
        torch.cuda.empty_cache()

        return FeatureCache(
            cls_tokens=cls_tokens,
            patch_tokens=patch_tokens,
            image_id_to_idx=image_id_to_idx,
        )

    def get_model_info(self) -> Dict[str, Dict[str, any]]:
        """Get information about loaded models."""
        info = {}
        for name, config in self.MODELS.items():
            if name in self.models:
                model = self.models[name]
                num_params = sum(p.numel() for p in model.parameters())
                info[name] = {
                    'config': config,
                    'num_parameters': num_params,
                    'device': next(model.parameters()).device,
                    'frozen': all(not p.requires_grad for p in model.parameters())
                }
        return info


def test_model_bank():
    """Quick test of ModelBank functionality."""
    print("Testing ModelBank...")
    
    # Create model bank
    bank = ModelBank(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print model info
    info = bank.get_model_info()
    for name, details in info.items():
        config = details['config']
        print(f"{name}: {details['num_parameters']:,} params, "
              f"{config.embed_dim}d, patch_size={config.patch_size}, "
              f"frozen={details['frozen']}")
    
    # Test feature extraction
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    if torch.cuda.is_available():
        dummy_images = dummy_images.cuda()
    
    # Test single model
    features = bank.get_features(dummy_images, ['dino'])
    print(f"DINOv2 features: patch_tokens {features['dino']['patch_tokens'].shape}, "
          f"cls_token {features['dino']['cls_token'].shape}")

    # Test all models
    features = bank.get_features(dummy_images, ['dino', 'siglip', 'mae'])
    for name, feats in features.items():
        print(f"{name}: patch_tokens {feats['patch_tokens'].shape}, "
              f"cls_token {feats['cls_token'].shape}")
    
    print("ModelBank test passed!")


if __name__ == "__main__":
    test_model_bank()