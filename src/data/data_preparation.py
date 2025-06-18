import os
from torch.utils.data import Dataset
from typing import Optional, List

class TextPromptDataset(Dataset):
    def __init__(self, file_path: str, transform: Optional[callable] = None):
        self.file_path = file_path
        self.transform = transform
        self.prompts = self._load_prompts(file_path)
    
    def _load_prompts(self, file_path: str) -> List[str]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        prompts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    prompts.append(clean_line)
        return prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        if self.transform:
            prompt = self.transform(prompt)
        return prompt
