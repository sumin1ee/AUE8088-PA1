# Python packages
from termcolor import colored
from typing import Dict
import copy
from torchvision.models import efficientnet_b1
# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

# Custom packages
from src.metric import MyAccuracy, MyF1ScoreConf, MyF1ScoreOvR
import src.config as cfg
from src.util import show_setting

class AddBatchNormWithSkip(AlexNet):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 → 32x32
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        self.downsample1 = nn.Conv2d(64, 192, kernel_size=1, stride=1)

        self.block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 → 16x16
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.downsample2 = nn.Conv2d(384, 256, kernel_size=1, stride=1)

        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 → 8x8
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        skip1 = self.downsample1(x1)
        x = x2 + skip1  # Skip connection after block2

        x = self.block3(x)
        x4 = self.block4(x)
        skip2 = self.downsample2(x)
        x = x4 + skip2  # Skip connection after block4

        x = self.block5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# [TODO: Optional] Rewrite this class if you want
class MyNetwork(AlexNet):
    def __init__(self):
        super().__init__()
        # [TODO] Modify feature extractor part in AlexNet
        self.reduced_receptive_fields_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),     # Output: 64×64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                    # Output: 32×32

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),   # Output: 32×32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                    # Output: 16×16

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),  # Output: 16×16
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # Output: 16×16
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Output: 16×16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)                     # Output: 8×8
        )

        self.add_batchnorm_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32×32

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 16×16

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 8×8
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [TODO: Optional] Modify this as well if you want
        x = self.reduced_receptive_fields_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork()
        elif model_name == 'AddBatchNormWithSkip':
            self.model = AddBatchNormWithSkip()
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            # only get the layer
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.accuracy = MyAccuracy()
        self.f1_score = MyF1ScoreOvR(num_classes=num_classes)

        # Hyperparameters
        self.save_hyperparameters()
    
    @staticmethod
    def _macro_average_f1_score(f1_scores: torch.Tensor) -> float:
        return f1_scores.mean().item()
    
    @staticmethod
    def _weighted_average_f1_score(f1_scores: torch.Tensor, weights: torch.Tensor) -> float:
        return (f1_scores * weights).sum().item() / weights.sum().item()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        # precision, recall, f1_score = self.f1_score(scores, y)
        # f1_score = self._macro_average_f1_score(f1_score)
        
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy, 'lr/train': self.optimizers().param_groups[0]['lr']},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        self.f1_score.update(scores, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy, 'lr/val': self.optimizers().param_groups[0]['lr']},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)
    
    def on_validation_epoch_end(self):
        f1_score = self.f1_score.compute()
        self.log_dict({'f1_score/val': f1_score}, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
            