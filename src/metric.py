from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.num_classes = num_classes
        self.tp = torch.zeros(num_classes)
        self.fp = torch.zeros(num_classes)
        self.fn = torch.zeros(num_classes)
    
    def update(self, preds, target):
        '''
        Args:
            preds: (B x C) tensor, where B is the batch size and C is the number of classes
            target: (B x C) tensor, where B is the batch size and C is the number of classes
        '''
        for i in range(self.num_classes):
            self.tp[i] = ((preds[:, i] == 1) & (target[:, i] == 1)).sum().item()
            self.fp[i] = ((preds[:, i] == 1) & (target[:, i] == 0)).sum().item()
            self.fn[i] = ((preds[:, i] == 0) & (target[:, i] == 1)).sum().item()
    
    def compute(self):
        '''
        Returns:
            f1_score: (1 x C) tensor, where C is the number of classes
        '''
        epsilon = 1e-10
        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon) 
        return f1_score

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = preds.argmax(dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape} and target {target.shape} must be the same")

        # [TODO] Cound the number of correct prediction
        correct = (preds == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()

