from typing import Tuple, List, Dict
from torch import nn, Tensor
from torch.utils.data import DataLoader
from modular.utils import exp_report

import torch
import os.path as osp
import os

class Trainer(nn.Module):
    def __init__(
            self, 
            model: nn.Module, 
            dataloader: Tuple[DataLoader, DataLoader], 
            loss_fn: nn.Module, 
            optimizer: torch.optim.Optimizer, 
            device: torch.device,
            epochs: int = 5,
    ) -> None:
        super(Trainer, self).__init__()
        self.model = model
        self.train_loader, self.test_loader = dataloader[0], dataloader[1]
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs

    def _train_step(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Tuple[float, float]:
    
        train_loss, train_acc = 0.0, 0.0
        model.train()

        for _, (data, labels) in enumerate(dataloader):

            data, labels = data.to(device), labels.to(device)

            out_logits = model(data)  # [32,2]

            loss = loss_fn(out_logits, labels)

            pred_labels = torch.max(torch.softmax(out_logits, dim=1), dim=1).indices  # [32]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (pred_labels == labels).sum().item() / len(labels) * 100.0

        return train_loss / len(dataloader), train_acc / len(dataloader)


    def _test_step(
            self,
            model: nn.Module,
            dataloader: DataLoader,
            loss_fn: nn.Module,
            device: torch.device,
    ) -> Tuple[float, float]:
        
        test_loss, test_acc = 0.0, 0.0
        model.eval()

        with torch.no_grad():
            for _, (data, labels) in enumerate(dataloader):

                data, labels = data.to(device), labels.to(device)

                out_logits = model(data)  # [32,2]

                loss = loss_fn(out_logits, labels)

                pred_labels = torch.max(torch.softmax(out_logits, dim=1), dim=1).indices  # [32]

                test_loss += loss.item()
                test_acc += (pred_labels == labels).sum().item() / len(labels) * 100.0

            return test_loss / len(dataloader), test_acc / len(dataloader)
        
    def forward(self) -> Dict[str, List[float]]:
        result = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
        test_best_acc = 0.0
        if not osp.exists("classifyproject/models"):
            os.makedirs("classifyproject/models")
        for epoch in range(self.epochs):

            train_loss, train_acc = self._train_step(
                model=self.model,
                dataloader=self.train_loader,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                device=self.device
            )

            test_loss, test_acc = self._test_step(
                model=self.model,
                dataloader=self.test_loader,
                loss_fn=self.loss_fn,
                device=self.device
            )
            result["train_loss"].append(train_loss)
            result["train_acc"].append(train_acc)
            result["test_loss"].append(test_loss)
            result["test_acc"].append(test_acc)

            exp_report(epoch, self.epochs, result)

            if test_acc > test_best_acc:
                test_best_acc = test_acc
                torch.save(self.model.state_dict(), "classifyproject/models/best_model.pth")

        return result
    

# trainer = Trainer(model=model, dataloader=(train_loader, test_loader), loss_fn=loss_fn, optimizer=optimizer, device=device, epochs=epochs)
