import torch
import open_clip

class BioCLIPModel:
    def __init__(self, model_name, pretrained_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.preprocess_train, self.preprocess_val = self._load_model(model_name, pretrained_path)

    def _load_model(self, model_name, pretrained_path):
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained_path
        )
        return model.to(self.device), preprocess_train, preprocess_val

    def generate_embeddings(self, image_tensors):
        """Generate embeddings for the preprocessed images."""
        with torch.no_grad(), torch.cuda.amp.autocast():
            embeddings = [self.model.encode_image(image_tensor.to(self.device)) for image_tensor in image_tensors]
            return torch.stack(embeddings, dim=0).squeeze()
