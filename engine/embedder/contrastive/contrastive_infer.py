import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from engine.embedder.contrastive.contrastive_model import XPrizeTreeEmbedder2NoDate, XPrizeTreeEmbedder, \
    XPrizeTreeEmbedder2
from engine.embedder.contrastive.contrastive_utils import ConditionalAutocast


def infer_model_without_labels(model, dataloader, device, use_mixed_precision, desc='Infering...', as_numpy=True):
    all_embeddings = []
    all_predicted_families = []

    for images, months, days in tqdm(dataloader, total=len(dataloader), desc=desc):
        embeddings, predicted_families = infer_batch(images=images, months=months, days=days, model=model,
                                                     device=device, use_mixed_precision=use_mixed_precision)

        all_embeddings.append(embeddings.detach().cpu())
        all_predicted_families.append(predicted_families)

    final_embeddings = torch.cat(all_embeddings, dim=0)
    final_predicted_families = sum(all_predicted_families, [])

    if as_numpy:
        final_embeddings = final_embeddings.numpy()

    return final_embeddings, final_predicted_families


def infer_model_with_labels(model, dataloader, device, use_mixed_precision, desc='Infering...', as_numpy=True):
    all_labels = []
    all_labels_ids = torch.tensor([])
    all_families = []
    all_families_ids = torch.tensor([])
    all_embeddings = torch.tensor([])
    all_predicted_families = []
    with torch.no_grad():
        for images, months, days, labels_ids, labels, families_ids, families in tqdm(dataloader, total=len(dataloader), desc=desc):
            embeddings, predicted_families = infer_batch(images=images, months=months, days=days, model=model,
                                                         device=device, use_mixed_precision=use_mixed_precision)

            all_labels.extend(labels)
            all_labels_ids = torch.cat((all_labels_ids, labels_ids.detach().cpu()), dim=0)
            all_families.extend(families)
            all_families_ids = torch.cat((all_families_ids, families_ids.detach().cpu()), dim=0)
            all_embeddings = torch.cat((all_embeddings, embeddings.detach().cpu()), dim=0)
            all_predicted_families.extend(predicted_families)

        if as_numpy:
            all_labels = np.array(all_labels)
            all_labels_ids = all_labels_ids.numpy()
            all_families = np.array(all_families)
            all_families_ids = all_families_ids.numpy()
            all_embeddings = all_embeddings.numpy()
            all_predicted_families = np.array(all_predicted_families)

        return all_labels, all_labels_ids, all_families, all_families_ids, all_embeddings, all_predicted_families


def infer_batch(images, months, days, model, device, use_mixed_precision):
    with torch.no_grad():
        data = torch.Tensor(images).to(device)
        months = torch.Tensor(months).to(device)
        days = torch.Tensor(days).to(device)
        if len(data.shape) == 3:
            data = data.unsqueeze(0)

        if isinstance(model, nn.DataParallel):
            actual_model = model.module
        else:
            actual_model = model

        with ConditionalAutocast(use_mixed_precision):
            if isinstance(actual_model, XPrizeTreeEmbedder2NoDate):
                output = model(data)
            else:
                output = model(data, months, days)

        if isinstance(actual_model, XPrizeTreeEmbedder):
            embeddings = output
            predicted_families = [None] * len(images)
        elif isinstance(actual_model, (XPrizeTreeEmbedder2, XPrizeTreeEmbedder2NoDate)):
            embeddings, classifier_logits = output[0], output[1]
            predicted_families_ids = torch.argmax(classifier_logits, dim=1)
            predicted_families = [actual_model.ids_to_families_mapping[int(family_id)] for family_id in predicted_families_ids]
        else:
            raise ValueError(f'Unknown model type: {actual_model.__class__}')

        return embeddings, predicted_families
