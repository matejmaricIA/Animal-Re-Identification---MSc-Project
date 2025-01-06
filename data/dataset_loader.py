from torch.utils.data import Dataset
from PIL import Image

class AnimalReIDDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = self.load_data()
        
    def load_data(self):
        # Load file paths and labels
        pass
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label