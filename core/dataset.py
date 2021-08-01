import torch

class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, emb_size, batch_size, n_batches, w_size=1):
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.w_size = w_size

    def __len__(self):
        return self.batch_size * self.n_batches

    def __getitem__(self, idx):
        noise = torch.randn(self.emb_size)
        if self.w_size > 1:
            noise = noise.unsqueeze(0).repeat(self.w_size, 1)
        return {"noise": noise}
