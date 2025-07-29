import torch
import torch.nn as nn
from transformers import BertModel

class LaciumFlowerModel(nn.Module):
    def __init__(self, pretrained_name="bert-base-multilingual-cased", hidden_size=768, num_tasks=3):
        super().__init__()
        self.encoder = BertModel.from_pretrained(pretrained_name)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.task_heads = nn.ModuleList([nn.Linear(hidden_size, 2) for _ in range(num_tasks)])

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        z = self.projection_head(cls_output)
        task_outputs = [head(cls_output) for head in self.task_heads]
        return z, task_outputs
