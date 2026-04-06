"""
GQA dataset loading and preprocessing for the Lewis superadditivity experiment.

Handles:
- Loading GQA from HuggingFace datasets
- Question type classification (spatial, semantic, material capabilities)
- Image preprocessing for multiple vision models
- Answer vocabulary construction
- Subset sampling for faster iteration
"""

import json
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import datasets


@dataclass
class GQAQuestion:
    """Single GQA question with metadata."""
    question_id: str
    image_id: str
    question: str
    answer: str
    question_type: str
    capabilities_required: Set[str]  # {"spatial", "semantic", "material"}


class QuestionClassifier:
    """Classify GQA questions by required capabilities."""
    
    def __init__(self):
        # Keywords that indicate spatial reasoning
        self.spatial_keywords = {
            "left", "right", "above", "below", "behind", "front", "next to", "beside",
            "on top of", "under", "over", "between", "around", "near", "far", "close",
            "inside", "outside", "in front of", "position", "where", "location"
        }
        
        # Keywords that indicate semantic/category reasoning  
        self.semantic_keywords = {
            "what", "who", "which", "animal", "person", "object", "thing", "type", "kind",
            "category", "name", "called", "species", "breed", "model", "brand", "style",
            "largest", "smallest", "biggest", "size", "count", "how many", "number"
        }
        
        # Keywords that indicate material/texture reasoning
        self.material_keywords = {
            "material", "made of", "texture", "surface", "shiny", "matte", "glossy",
            "rough", "smooth", "metal", "metallic", "wood", "wooden", "plastic", 
            "glass", "fabric", "leather", "stone", "concrete", "brick", "paper",
            "transparent", "opaque", "reflective", "color", "colored"
        }
    
    def classify_question(self, question: str, question_type: str = None) -> Set[str]:
        """Classify which capabilities a question requires.
        
        Args:
            question: The question text
            question_type: Optional GQA question type annotation
            
        Returns:
            Set of required capabilities: {"spatial", "semantic", "material"}
        """
        question_lower = question.lower()
        capabilities = set()
        
        # Check for spatial indicators
        if any(keyword in question_lower for keyword in self.spatial_keywords):
            capabilities.add("spatial")
            
        # Check for semantic indicators
        if any(keyword in question_lower for keyword in self.semantic_keywords):
            capabilities.add("semantic")
            
        # Check for material indicators  
        if any(keyword in question_lower for keyword in self.material_keywords):
            capabilities.add("material")
            
        # Use GQA's question type as additional signal
        if question_type:
            qtype_lower = question_type.lower()
            if any(word in qtype_lower for word in ["spatial", "relate", "position"]):
                capabilities.add("spatial")
            if any(word in qtype_lower for word in ["category", "attribute", "object"]):
                capabilities.add("semantic")
                
        # Default: if no specific indicators, assume semantic (most questions ask "what")
        if not capabilities:
            capabilities.add("semantic")
            
        return capabilities


class GQADataset(Dataset):
    """GQA dataset with question capability classification."""
    
    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        answer_vocab: Optional[Dict[str, int]] = None,
        vocab_size: int = 1500
    ):
        """Initialize GQA dataset.

        Args:
            split: Dataset split ("train", "val", or "test")
            max_samples: Limit dataset size (for faster iteration)
            answer_vocab: Pre-built answer vocabulary
            vocab_size: Size of answer vocabulary to build
        """
        self.split = split
        self.max_samples = max_samples
        self.vocab_size = vocab_size
        self.classifier = QuestionClassifier()
        
        # Load dataset from HuggingFace
        # GQA has separate configs for instructions (Q&A) and images
        # The HF split name for "test" is actually "testdev"
        hf_split_map = {
            "train": "train",
            "val": "val",
            "test": "testdev",
        }
        hf_split = hf_split_map.get(split, split)

        instr_config = f"{hf_split}_balanced_instructions"
        image_config = f"{hf_split}_balanced_images"

        print(f"Loading GQA {split} split (instructions: {instr_config})...")
        self.dataset = datasets.load_dataset("lmms-lab/GQA", instr_config, split=hf_split)

        print(f"Loading GQA {split} images (config: {image_config})...")
        images_ds = datasets.load_dataset("lmms-lab/GQA", image_config, split=hf_split)

        # Build image lookup: imageId -> PIL Image
        print("Building image lookup table...")
        self.image_lookup = {}
        for item in images_ds:
            self.image_lookup[str(item['id'])] = item['image']
        print(f"Loaded {len(self.image_lookup)} unique images")

        # Sample subset if requested
        if max_samples and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))
            print(f"Sampled {max_samples} examples from {split} split")
        
        # Build or use provided answer vocabulary
        if answer_vocab is None and split == "train":
            self.answer_vocab = self._build_answer_vocab()
        else:
            self.answer_vocab = answer_vocab or {}
            
        # Process questions and classify capabilities
        self.questions = self._process_questions()

        print(f"Loaded {len(self.questions)} questions from GQA {split}")
        self._print_capability_stats()
    
    def _build_answer_vocab(self) -> Dict[str, int]:
        """Build answer vocabulary from training data."""
        print("Building answer vocabulary...")
        answer_counts = Counter()
        
        for item in self.dataset:
            answer = str(item['answer']).lower().strip()
            answer_counts[answer] += 1
        
        # Keep top-K most frequent answers
        top_answers = answer_counts.most_common(self.vocab_size - 1)  # -1 for <UNK>
        vocab = {"<UNK>": 0}
        for i, (answer, count) in enumerate(top_answers):
            vocab[answer] = i + 1
            
        print(f"Built vocabulary with {len(vocab)} answers (top {self.vocab_size})")
        return vocab
    
    def _process_questions(self) -> List[GQAQuestion]:
        """Process raw dataset into GQAQuestion objects."""
        questions = []
        
        for item in self.dataset:
            question_id = str(item.get('id', len(questions)))
            image_id = str(item.get('imageId', ''))
            question_text = str(item['question'])
            answer = str(item['answer']).lower().strip()
            # 'types' is a dict like {'structural': 'verify', 'semantic': 'attr', 'detailed': 'verifyAttr'}
            types_dict = item.get('types', {})
            question_type = types_dict.get('detailed', '') if isinstance(types_dict, dict) else str(types_dict)
            
            # Classify required capabilities
            capabilities = self.classifier.classify_question(question_text, question_type)
            
            questions.append(GQAQuestion(
                question_id=question_id,
                image_id=image_id,
                question=question_text,
                answer=answer,
                question_type=question_type,
                capabilities_required=capabilities
            ))
        
        return questions
    
    def _print_capability_stats(self):
        """Print statistics about capability requirements."""
        capability_counts = {
            "spatial": 0,
            "semantic": 0, 
            "material": 0,
            "spatial+semantic": 0,
            "spatial+material": 0,
            "semantic+material": 0,
            "all_three": 0
        }
        
        for q in self.questions:
            caps = q.capabilities_required
            if caps == {"spatial"}:
                capability_counts["spatial"] += 1
            elif caps == {"semantic"}:
                capability_counts["semantic"] += 1
            elif caps == {"material"}:
                capability_counts["material"] += 1
            elif caps == {"spatial", "semantic"}:
                capability_counts["spatial+semantic"] += 1
            elif caps == {"spatial", "material"}:
                capability_counts["spatial+material"] += 1
            elif caps == {"semantic", "material"}:
                capability_counts["semantic+material"] += 1
            elif caps == {"spatial", "semantic", "material"}:
                capability_counts["all_three"] += 1
        
        print("\nCapability requirements:")
        total = len(self.questions)
        for cap_type, count in capability_counts.items():
            pct = 100 * count / total if total > 0 else 0
            print(f"  {cap_type}: {count} ({pct:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single example as (image_tensor, answer_idx).

        Returns a single image tensor (224x224, ImageNet-normalized) and the
        answer class index.  Per-model preprocessing happens inside ModelBank.
        """
        question = self.questions[idx]

        # Load image from lookup table by imageId
        image = self.image_lookup.get(question.image_id)
        if image is None:
            image = Image.new('RGB', (224, 224))
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        image = image.convert('RGB')

        # Use a single shared transform; per-model normalization is in ModelBank
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image)

        answer_idx = self.answer_vocab.get(question.answer, 0)  # 0 = <UNK>
        return image_tensor, torch.tensor(answer_idx, dtype=torch.long)
    
    def get_answer_vocab(self) -> Dict[str, int]:
        """Get the answer vocabulary."""
        return self.answer_vocab
    
    def get_capability_subsets(self) -> Dict[str, List[int]]:
        """Get indices of questions requiring specific capability combinations."""
        subsets = {
            "spatial_only": [],
            "semantic_only": [], 
            "material_only": [],
            "spatial_semantic": [],
            "spatial_material": [],
            "semantic_material": [],
            "all_three": [],
            "two_or_more": [],
            "exactly_two": []
        }
        
        for i, question in enumerate(self.questions):
            caps = question.capabilities_required
            
            if caps == {"spatial"}:
                subsets["spatial_only"].append(i)
            elif caps == {"semantic"}:
                subsets["semantic_only"].append(i)
            elif caps == {"material"}:
                subsets["material_only"].append(i)
            elif caps == {"spatial", "semantic"}:
                subsets["spatial_semantic"].append(i)
                subsets["two_or_more"].append(i)
                subsets["exactly_two"].append(i)
            elif caps == {"spatial", "material"}:
                subsets["spatial_material"].append(i)
                subsets["two_or_more"].append(i)
                subsets["exactly_two"].append(i)
            elif caps == {"semantic", "material"}:
                subsets["semantic_material"].append(i)
                subsets["two_or_more"].append(i)
                subsets["exactly_two"].append(i)
            elif caps == {"spatial", "semantic", "material"}:
                subsets["all_three"].append(i)
                subsets["two_or_more"].append(i)
        
        return subsets


def create_gqa_dataloaders(
    train_samples: Optional[int] = None,
    eval_samples: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Create GQA train and validation dataloaders.

    Args:
        train_samples: Limit training samples (None = use all)
        eval_samples: Limit eval samples (None = use all)
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers

    Returns:
        (train_loader, val_loader, answer_vocab)
    """
    # Create training dataset and build vocabulary
    train_dataset = GQADataset(
        split="train",
        max_samples=train_samples,
    )

    # Create validation dataset with same vocabulary
    val_dataset = GQADataset(
        split="val",
        max_samples=eval_samples,
        answer_vocab=train_dataset.get_answer_vocab(),
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.get_answer_vocab()


def analyze_question_capabilities(dataset: GQADataset, save_path: Optional[str] = None):
    """Analyze and optionally save question capability statistics."""
    stats = {
        "total_questions": len(dataset),
        "capability_combinations": {},
        "individual_capabilities": {"spatial": 0, "semantic": 0, "material": 0},
        "sample_questions": {}
    }
    
    # Count capability combinations
    for question in dataset.questions:
        caps = question.capabilities_required
        caps_key = "+".join(sorted(caps))
        
        if caps_key not in stats["capability_combinations"]:
            stats["capability_combinations"][caps_key] = 0
        stats["capability_combinations"][caps_key] += 1
        
        # Count individual capabilities
        for cap in caps:
            stats["individual_capabilities"][cap] += 1
            
        # Collect sample questions
        if caps_key not in stats["sample_questions"]:
            stats["sample_questions"][caps_key] = []
        if len(stats["sample_questions"][caps_key]) < 3:
            stats["sample_questions"][caps_key].append(question.question)
    
    # Print analysis
    print("\nDetailed Capability Analysis:")
    print("-" * 40)
    for combo, count in sorted(stats["capability_combinations"].items()):
        pct = 100 * count / stats["total_questions"]
        print(f"{combo}: {count} ({pct:.1f}%)")
        
        # Show sample questions
        if combo in stats["sample_questions"]:
            for i, q in enumerate(stats["sample_questions"][combo][:2]):
                print(f"  Example {i+1}: {q}")
        print()
    
    # Save to file if requested
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Analysis saved to {save_path}")
    
    return stats


if __name__ == "__main__":
    # Test the dataset loading
    print("Testing GQA dataset loading...")
    
    train_loader, val_loader, vocab = create_gqa_dataloaders(
        train_samples=1000,  # Small sample for testing
        eval_samples=200,
        batch_size=8
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Answer vocab size: {len(vocab)}")
    
    # Test a batch
    images, answers = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Answer batch shape: {answers.shape}")
    
    # Analyze capabilities
    train_dataset = train_loader.dataset
    analyze_question_capabilities(train_dataset)