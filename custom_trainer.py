import torch
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def __init__(self,model,
        args,
        train_dataset,
        eval_dataset,
        compute_metrics,
        weights):
        super(CustomTrainer, self).__init__(model=model, args=args,train_dataset=train_dataset, eval_dataset=eval_dataset,compute_metrics=compute_metrics)
        self.weights = weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device=self.model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss