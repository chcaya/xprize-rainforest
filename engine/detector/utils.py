from shapely import box
from torch.optim.lr_scheduler import StepLR


def detector_result_to_lists(detector_result):
    detector_result = [{k: v.cpu().numpy() for k, v in x.items()} for x in detector_result]
    for x in detector_result:
        x['boxes'] = [box(*b) for b in x['boxes']]
        x['scores'] = x['scores'].tolist()
    boxes = [x['boxes'] for x in detector_result]
    scores = [x['scores'] for x in detector_result]

    return boxes, scores


class WarmupStepLR:
    def __init__(self, optimizer, step_size, gamma=0.1, warmup_steps=10, base_lr=1e-6):
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.max_lr = optimizer.param_groups[0]['lr']
        self.optimizer = optimizer
        self.scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        self.current_step = 0
        self.step()

    def step(self):
        # Warm-up phase
        if self.current_step < self.warmup_steps + 1:
            lr = self.base_lr + (self.max_lr - self.base_lr) * self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        # StepLR phase
        else:
            self.scheduler.step()
        self.current_step += 1

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

