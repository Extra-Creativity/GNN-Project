import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import scipy
from data import load_data
from tqdm import tqdm
import torch.nn.functional as F
import multiprocessing

def top_k_eigenvector(x: np.ndarray, idx: int, k: int):
    y = x - x[idx]
    y = np.concat([y[:idx], y[idx + 1:]])
    mat = y @ y.T
    eigen_val, eigen_vec = scipy.linalg.eigh(mat, subset_by_index=[0, k-1])
    # if scale isn't considered, eigen_val could be stripped out.
    return np.concat([eigen_val.reshape((1, *eigen_val.shape)), eigen_vec])

def top_k_eigenvalue(x: np.ndarray, idx: int, k: int):
    y = x - x[idx]
    y = np.concat([y[:idx], y[idx + 1:]])
    mat = y @ y.T
    eigen_val = scipy.linalg.eigh(mat, subset_by_index=[0, k-1], eigvals_only=True)
    return eigen_val

class ModelNet40_AutoEnc(Dataset):
    def __init__(self, num_points, k, data_size=None, partition='train'):
        super().__init__()
        self.data, _ = load_data(partition)
        if data_size is not None:
            self.data = self.data[:data_size]
        self.num_points = num_points
        self.k = k

    def __getitem__(self, item):
        pointcloud_id, point_id = item // self.num_points, item % self.num_points
        pointcloud = self.data[pointcloud_id][:self.num_points]
        feature = top_k_eigenvector(pointcloud, point_id, self.k)
        return torch.tensor(feature.T)

    def __len__(self):
        return self.data.shape[0] * self.num_points

class PointFeatureAutoEncoder(nn.Module):
    def __init__(self, input_channels, output_channels=40):
        super().__init__()
        self.dl1 = nn.Linear(input_channels, input_channels // 2)
        self.dl2 = nn.Linear(input_channels // 2, input_channels // 4)
        self.dl3 = nn.Linear(input_channels // 4, input_channels // 8)
        self.dl4 = nn.Linear(input_channels // 8, output_channels)

        self.ul4 = nn.Linear(output_channels, input_channels // 8)
        self.ul3 = nn.Linear(input_channels // 8, input_channels // 4)
        self.ul2 = nn.Linear(input_channels // 4, input_channels // 2)
        self.ul1 = nn.Linear(input_channels // 2, input_channels)

        self.canonical = nn.Conv1d(15, 3, kernel_size=1)

    def forward(self, x):
        x0 = F.relu(self.dl1(x))
        x1 = F.relu(self.dl2(x0))
        x2 = F.relu(self.dl3(x1))
        x3 = F.relu(self.dl4(x2))

        y3 = F.relu(torch.concat((self.ul4(x3), x2)))
        y2 = F.relu(torch.concat((self.ul3(y3), x1)))
        y1 = F.relu(torch.concat((self.ul2(y2), x0)))
        y0 = F.relu(torch.concat((self.ul1(y1), x)))

        result = self.canonical(y0)
        return result

device = "cuda"

def train(epochs, model, train_loader, test_loader, criterion, opt, scheduler):
    best_test_loss = 100000000.0
    for epoch in range(epochs):
        train_loss = 0.0
        count = 0.0
        model.train()
        for data in tqdm(train_loader):
            data = data.to(device)
            batch_size = data.size()[0]
            opt.zero_grad()
            result = model(data)
            loss = criterion(result, data)
            if loss.isnan():
                print(torch.isnan(data).any())
                print(result)
                assert False

            loss.backward()
            opt.step()
            train_loss += loss.item() * batch_size
            count = count + 1
        scheduler.step()
        print('Train %d, loss: %.6f' % (epoch, train_loss*1.0/count))

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        with torch.no_grad():
            for data in tqdm(test_loader):
                data = data.to(device)
                batch_size = data.size()[0]
                result = model(data)
                loss = criterion(result, data)
                test_loss += loss.item() * batch_size
                count = count + 1

            print('Test %d, loss: %.6f' % (epoch, test_loss*1.0/count))
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), 'checkpoints/autoenc/model.t7')

def run():
    num_points = 1024
    train_dataset = ModelNet40_AutoEnc(num_points, 3, 1024)
    test_dataset = ModelNet40_AutoEnc(num_points, 3, 10, "test")
    model = PointFeatureAutoEncoder(num_points, 16).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)

    train(10, model, train_dataset, test_dataset, F.mse_loss, optim, scheduler)

def to_eigenvals(arg):
    num_points = 1024
    k = 3

    dataset, begin, size = arg
    final_result = []
    end = min(len(dataset[0]), begin + size)

    for data in tqdm(dataset[0][begin:end]):
        real_data = data[:num_points]
        # print(real_data.shape)
        result_data = np.ndarray((*real_data.shape[:-1], k))
        for pointid in range(real_data.shape[0]):
            eigen_values = top_k_eigenvalue(real_data, pointid, k)
            result_data[pointid] = eigen_values
        # print(result_data.shape)
        final_result.append(result_data)
    return final_result

def run_only_eigenvalue():
    train_dataset = load_data("train")
    test_dataset = load_data("test")

    size = 1000
    def parallel_to_eigenvals(dataset, save_path):
        tasks = []
        begin = 0
        while begin < len(dataset[0]):
            tasks.append(begin)
            begin += size

        task_num = len(tasks)
        with multiprocessing.Pool(min(2, task_num)) as p:
            result = p.map(to_eigenvals, [(dataset, task, size) for task in tasks])
            final_result = []
            for r in result:
                final_result.extend(r)

            final_result = np.array(final_result)
            np.savez_compressed(save_path,feature=final_result)

    parallel_to_eigenvals(train_dataset, "data/eigenvalues_train.npz")
    parallel_to_eigenvals(test_dataset, "data/eigenvalues_test.npz")

if __name__ == "__main__":
    run_only_eigenvalue()