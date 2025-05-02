from torchmetrics import Metric
import torch

class MyF1ScoreConf(Metric):
    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.num_classes = num_classes
        self.add_state('conf_matrix', default=torch.zeros(num_classes, num_classes), dist_reduce_fx='sum')
    
    def update(self, preds, target):
        '''
        Args:
            preds: (B, C) tensor, where B is the batch size and C is the number of classes
            target: (B, ) tensor
        
        [WARNING} NOTE: update() is called each iteration, so the metric should be updated in-place
        '''
        preds = preds.argmax(dim=1)
        
        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape} and target {target.shape} must be the same")
        
        # [TODO] Compute confusion matrix
        for t, p in zip(target, preds):
            self.conf_matrix[t, p] += 1
    
    def compute(self):
        '''
        Returns:
            precision: (C, ) tensor, where C is the number of classes
            recall: (C, ) tensor, where C is the number of classes
            f1_score: (C, ) tensor, where C is the number of classes
            
        [WARNING] NOTE: compute() is called at the end of each epoch
        '''
        epsilon = 1e-10
        tp = torch.diag(self.conf_matrix)
        fp = self.conf_matrix.sum(dim=0) - tp
        fn = self.conf_matrix.sum(dim=1) - tp
        
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon) 
        return precision, recall, f1_score

# [TODO] Implement this!
class MyF1ScoreOvR(Metric):
    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.num_classes = num_classes
        self.add_state('tp', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros(num_classes), dist_reduce_fx='sum')
    
    def update(self, preds, target):
        '''
        Args:
            preds: (B, C) tensor, where B is the batch size and C is the number of classes
            target: (B, ) tensor
        
        [WARNING} NOTE: update() is called each iteration, so the metric should be updated in-place
        '''
        preds = preds.argmax(dim=1)
        
        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape} and target {target.shape} must be the same")
        
        # [TODO] Compute true positives, false positives, and false negatives
        
        for i in range(self.num_classes):
            self.tp[i] += ((preds == i) & (target == i)).sum().float()
            self.fp[i] += ((preds == i) & (target != i)).sum().float()
            self.fn[i] += ((preds != i) & (target == i)).sum().float()
        
    
    def compute(self):
        '''
        Returns:
            f1_score: (C, ) tensor, where C is the number of classes
            
        [WARNING] NOTE: compute() is called at the end of each epoch
        '''
        epsilon = 1e-10
        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon) 
        return f1_score.mean().item()

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
    
def main():
    preds = torch.tensor([
        [0.1, 0.9, 0.0], # 1
        [0.8, 0.1, 0.1], # 0
        [0.3, 0.2, 0.5], # 2
        [0.0, 0.2, 0.8], # 2
        [0.6, 0.3, 0.1], # 0
    ]) # (1, 0, 2, 2, 0)

    target = torch.tensor([1, 0, 1, 2, 0])

    num_classes = 3

    acc = MyAccuracy()
    f1_confmat = MyF1ScoreConf(num_classes)
    f1_manual = MyF1ScoreOvR(num_classes)

    # update
    acc.update(preds, target)
    f1_confmat.update(preds, target)
    f1_manual.update(preds, target)

    # compute
    accuracy = acc.compute()
    precision, recall, f1_conf = f1_confmat.compute()
    f1_v2 = f1_manual.compute()
    
    macro_f1_score = f1_conf.mean().item()

    print("=== Accuracy ===")
    print(f"Accuracy: {accuracy.item():.4f}\n")

    print("=== MyF1Score (CM-based) ===")
    print(f"Precision: {precision}")
    print(f"Recall   : {recall}")
    print(f"F1 Score : {f1_conf}\n")
    print(f'Macro F1 Score: {macro_f1_score:.4f}\n')

    print("=== MyF1ScoreV2 (OvR based) ===")
    print(f"F1 Score : {f1_v2}\n")

if __name__ == "__main__":
    main()

