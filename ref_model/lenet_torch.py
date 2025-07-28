import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm

# LeNet-5 结构
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

def load_numpy_dataset(data_dir):
    # 加载和预处理数据，假设数据shape和你的numpy实现一致
    def load_images(path):
        with open(path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows, cols)
            return images
    def load_labels(path):
        with open(path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num = int.from_bytes(f.read(4), 'big')
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
    train_images = load_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz').replace('.gz',''))
    train_labels = load_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz').replace('.gz',''))
    test_images = load_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz').replace('.gz',''))
    test_labels = load_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz').replace('.gz',''))
    return train_images, train_labels, test_images, test_labels

def load_mnist(data_dir):
    # 如果是.gz格式，先解压
    import gzip
    def extract_gz(src):
        if src.endswith('.gz'):
            dst = src[:-3]
            if not os.path.exists(dst):
                with gzip.open(src, 'rb') as f_in, open(dst, 'wb') as f_out:
                    f_out.write(f_in.read())
            return dst
        return src
    for fname in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
        extract_gz(os.path.join(data_dir, fname))
    return load_numpy_dataset(data_dir)

def preprocess(images):
    # [N, 28, 28] -> [N, 1, 32, 32], 归一化到[0,1]
    images = images.astype(np.float32) / 255.0
    N = images.shape[0]
    padded = np.zeros((N, 32, 32), dtype=np.float32)
    padded[:, 2:30, 2:30] = images
    return padded[:, None, :, :]

def train_and_eval(data_dir, epochs=10, batch_size=64, lr=0.01, device='cpu'):
    train_images, train_labels, test_images, test_labels = load_mnist(data_dir)
    X_train = preprocess(train_images)
    X_test = preprocess(test_images)
    y_train = train_labels
    y_test = test_labels
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    model = LeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    if epochs == 0:
        return model, test_loader
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            pbar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs} - train acc: {train_acc:.4f} - loss: {avg_loss:.4f} - test acc: {test_acc:.4f}")
    print("Final test accuracy:", test_acc)
    return model, test_loader

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")
    return model
def evaluate(model, loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Testing", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    print(f"Test set: {correct}/{total} correct, accuracy: {correct/total:.4f}")
    return correct / total

def infer(model, image):
    # image: numpy array [1, 32, 32] or [32, 32]
    if image.ndim == 2:
        image = image[None, :, :]
    image = image.astype(np.float32) / 255.0
    padded = np.zeros((1, 32, 32), dtype=np.float32)
    padded[:, 2:30, 2:30] = image
    x = torch.tensor(padded[None, ...])  # [1,1,32,32]
    model.eval()
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)
        pred = prob.argmax(dim=1).item()
        conf = prob.max().item()
        print("Class probabilities:")
        for i, p in enumerate(prob[0].tolist()):
            print(f"  class {i}: {p:.4f}")
    return pred, conf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../dataset')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=str, choices=['train', 'infer', 'test'], default='train')
    parser.add_argument('--image', type=str, help='单张图片推理')
    parser.add_argument('--model-path', type=str, default='ref_model/lenet.pth', help='模型权重保存/加载路径')
    args = parser.parse_args()
    if args.mode == 'train':
        model, _ = train_and_eval(args.data_dir, args.epochs, args.batch_size, args.lr, args.device)
        save_model(model, args.model_path)
    elif args.mode == 'test':
        model = LeNet().to(args.device)
        load_model(model, args.model_path, args.device)
        _, test_loader = train_and_eval(args.data_dir, 0, args.batch_size, args.lr, args.device)
        # 用加载的模型评估
        acc = evaluate(model, test_loader, args.device)
        print(f"Loaded model test accuracy: {acc:.4f}")
    elif args.mode == 'infer':
        model = LeNet().to(args.device)
        load_model(model, args.model_path, args.device)
        from PIL import Image
        img = Image.open(args.image).convert('L').resize((28,28))
        img_np = np.array(img)
        pred, conf = infer(model, img_np)
        print(f"Predicted: {pred}, Confidence: {conf:.4f}")

"""
python ref_model/lenet_torch.py --mode train --data-dir dataset --epochs 1 --batch-size 64 --lr 0.01

python ref_model/lenet_torch.py --mode test --data-dir dataset --batch-size 64

python ref_model/lenet_torch.py --mode infer --data-dir dataset --image your_image.png

"""
