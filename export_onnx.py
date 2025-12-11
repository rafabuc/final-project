import torch
import os
import sys
from hydra import compose, initialize
from omegaconf import OmegaConf

# Python found modules src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.multimodal_net import MultimodalNet

# --- CONFIG ---
CHECKPOINT_PATH = "src/training/checkpoints/best_model_1.7.pth" # local path
OUTPUT_ONNX_PATH = "multimodal_model.onnx"

# 1. Define architecture (Must match training)
embedding_dim = 512
fusion_hidden_dim = 256
dropout_rate = 0.0 # Dropout in 0 for export

# 2. Initialize model
device = 'cpu'
model = MultimodalNet(
    embedding_dim=embedding_dim,
    fusion_hidden_dim=fusion_hidden_dim,
    dropout_rate=dropout_rate,
    freeze_vision=False,
    freeze_text=False
)

# 3. Load weights
print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Remove module prefixes if exist
new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

model.eval() # Evaluation mode

# 4. Create ONNX Wrapper
# ONNX prefers separated inputs over dictionaries
class OnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, image, input_ids, attention_mask):
        # Reconstruct the dictionary that MultimodalNet expects
        batch = {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        return self.model(batch)

wrapped_model = OnnxWrapper(model)

# 5. Create Dummy Inputs (Fake data for tracing the graph)
print("Generating ONNX graph...")
dummy_image = torch.randn(1, 3, 224, 224)
dummy_input_ids = torch.randint(0, 1000, (1, 128))
dummy_mask = torch.ones(1, 128).long()

# 6. Export
torch.onnx.export(
    wrapped_model,
    (dummy_image, dummy_input_ids, dummy_mask),
    OUTPUT_ONNX_PATH,
    export_params=True,
    opset_version=18, # Stable version
    do_constant_folding=True,
    input_names=['image', 'input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'image': {0: 'batch_size'},
        'input_ids': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    }
)

print(f"✅ Success! Model exported to: {OUTPUT_ONNX_PATH}")