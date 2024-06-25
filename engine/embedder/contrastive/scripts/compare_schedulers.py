import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts

if __name__ == "__main__":
    # Define a simple model
    model = torch.nn.Linear(10, 1)
    optimizer1 = optim.AdamW(model.parameters(), lr=0.001)
    optimizer2 = optim.AdamW(model.parameters(), lr=0.001)
    optimizer3 = optim.AdamW(model.parameters(), lr=0.001)

    # Define schedulers
    cosine_scheduler = CosineAnnealingLR(optimizer1, T_max=10)
    cosine_scheduler_warm = CosineAnnealingWarmRestarts(optimizer2, T_0=50)
    step_scheduler = StepLR(optimizer3, step_size=30, gamma=0.5)

    # Lists to store learning rates
    cosine_lrs = []
    cosine_warm_lrs = []
    step_lrs = []

    # Simulate training for 100 epochs
    for epoch in range(100):
        # Update learning rates
        cosine_scheduler.step()
        cosine_scheduler_warm.step()
        step_scheduler.step()

        # Store learning rates
        cosine_lrs.append(cosine_scheduler.get_last_lr()[0])
        cosine_warm_lrs.append(cosine_scheduler_warm.get_last_lr()[0])
        step_lrs.append(step_scheduler.get_last_lr()[0])

    # Plot learning rates
    plt.figure(figsize=(10, 6))
    plt.plot(cosine_lrs, label='CosineAnnealingLR', marker='o')
    plt.plot(cosine_warm_lrs, label='CosineAnnealingWarmRestarts', marker='o')
    plt.plot(step_lrs, label='StepLR', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
