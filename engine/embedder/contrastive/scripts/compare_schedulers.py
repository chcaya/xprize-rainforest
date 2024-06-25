import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts, \
    ReduceLROnPlateau, LRScheduler
from warmup_scheduler import GradualWarmupScheduler

if __name__ == "__main__":
    # Define a simple model
    model = torch.nn.Linear(10, 1)
    lr = 0.001
    warmup_period = 3
    optimizer1 = optim.AdamW(model.parameters(), lr=lr)
    optimizer2 = optim.AdamW(model.parameters(), lr=lr)
    optimizer3 = optim.AdamW(model.parameters(), lr=lr)
    optimizer4 = optim.AdamW(model.parameters(), lr=lr)

    # Define schedulers
    cosine_scheduler = CosineAnnealingLR(optimizer1, T_max=10)
    cosine_scheduler_warm = CosineAnnealingWarmRestarts(optimizer2, T_0=10)
    cosine_scheduler_for_warmup = CosineAnnealingLR(optimizer3, T_max=10)
    cosine_scheduler_warmup = GradualWarmupScheduler(optimizer3, multiplier=1, total_epoch=5, after_scheduler=cosine_scheduler_for_warmup)
    step_scheduler = StepLR(optimizer4, step_size=30, gamma=0.5)

    # Lists to store learning rates
    cosine_lrs = []
    cosine_warm_lrs = []
    warmup_lrs = []
    cosine_warmup_lrs = []
    step_lrs = []

    # Simulate training for 100 epochs
    for epoch in range(100):
        # Update learning rates
        cosine_scheduler.step()
        cosine_scheduler_warm.step()
        cosine_scheduler_warmup.step()
        step_scheduler.step()

        # Store learning rates
        cosine_lrs.append(cosine_scheduler.get_last_lr()[0])
        cosine_warm_lrs.append(cosine_scheduler_warm.get_last_lr()[0])
        cosine_warmup_lrs.append(cosine_scheduler_warmup.get_last_lr()[0])
        step_lrs.append(step_scheduler.get_last_lr()[0])

    # Plot learning rates
    plt.figure(figsize=(10, 6))
    plt.plot(cosine_lrs, label='CosineAnnealingLR', marker='o')
    plt.plot(cosine_warm_lrs, label='CosineAnnealingWarmRestarts', marker='o')
    plt.plot(cosine_warmup_lrs, label='CosineAnnealingLR with warmup', marker='o')
    plt.plot(step_lrs, label='StepLR', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
