from sentence_transformers import SentenceTransformer
from config import settings
import torch

_model = None
_device = None

def get_device():
    global _device
    if _device is None:
        # Priority 1 — NVIDIA CUDA
        try:
            import torch
            if torch.cuda.is_available():
                _device = "cuda"
                print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
                return _device
        except ImportError:
            pass

        # Priority 2 — AMD DirectML
        try:
            import torch_directml
            _device = torch_directml.device()
            print("Using DirectML (AMD GPU)")
            return _device
        except ImportError:
            pass

        # Priority 3 — CPU fallback
        _device = "cpu"
        print("No GPU found, using CPU")

    return _device

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            settings.embed_model,
            device=get_device(),
        )
    return _model
    
def embed_text(text: str) -> list[float]:
    model = get_model()
    return model.encode(text, normalize_embeddings=True).tolist()

# def embed_batch(texts: list[str]) -> list[list[float]]:
#     model = get_model()

#     # Larger batch size for GPU, smaller for CPU
#     device = get_device()
#     batch_size = 128 if str(device) != "cpu" else 32

#     return model.encode(
#         texts,
#         normalize_embeddings=True,
#         batch_size=batch_size,
#     ).tolist()

def embed_batch(texts: list[str]) -> list[list[float]]:
    import torch
    import torch.nn.functional as F

    model = get_model()
    tokenizer = model.tokenizer
    transformer = model[0].auto_model  # underlying HuggingFace model

    with torch.no_grad():
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(get_device())

        output = transformer(**encoded)
        # Mean pooling
        attention_mask = encoded['attention_mask']
        token_embeddings = output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().tolist()


