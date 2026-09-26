import torch
import numpy

class HeartFailureDataset(torch.utils.data.Dataset):

    def __init__(self, filepath, rows):
        self.data = numpy.loadtxt(filepath, skiprows=1, delimiter=',')
        self.data = self.data[rows, :]
        self.x_data = self.data[:, :-1]
        
        x_min = numpy.min(self.x_data, axis=0)
        x_max = numpy.max(self.x_data, axis=0)
        
        self.x_data = (self.x_data - x_min) / (x_max - x_min)
        self.x_data = torch.tensor(self.x_data, dtype=torch.float32)

        self.y_data = torch.tensor(self.data[:, -1], dtype=torch.long)

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index, :]
        y = self.y_data[index]

        return x, y


class HeartFailureNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(12, 64)
        self.relu1 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(64, 16)
        self.relu2 = torch.nn.ReLU()

        self.fcout = torch.nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fcout(x)

        return x

# Create dataset instances
dataset_filepath = r"C:\Users\Matthew\Downloads\heart_failure_clinical_records_dataset.csv"

train_rows = range(0, 200)
val_rows = range(200, 250)
test_rows = range(250, 299)

train_dataset = HeartFailureDataset(dataset_filepath, train_rows)
val_dataset = HeartFailureDataset(dataset_filepath, val_rows)
test_dataset = HeartFailureDataset(dataset_filepath, test_rows)

batch_size = 32

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_rows), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_rows), shuffle=False)


heart_failure_network = HeartFailureNetwork()
print(heart_failure_network)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(heart_failure_network.parameters(), lr=0.001)

batches_per_epoch = int(len(train_dataset) / batch_size)

num_epochs = 10
for epoch in range(num_epochs):
    print("Epoch: " + str(epoch))
    heart_failure_network.train()
    for batch_x, batch_y in train_loader:
        batch_y_pred = heart_failure_network(batch_x)

        loss = criterion(batch_y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print("Batch loss: " + str(loss))
    print("Last train batch loss: " + str(loss))

    heart_failure_network.eval()
    for batch_x, batch_y in val_loader:
        batch_y_pred = heart_failure_network(batch_x)    
        loss = criterion(batch_y_pred, batch_y)

        print("Val loss: " + str(loss))

        # print(numpy.argmax(batch_y_pred.detach(), axis=1))
        # print(batch_y)
        accuracy = numpy.sum((numpy.argmax(batch_y_pred.detach(), axis=1) == batch_y.detach()).numpy()) / len(batch_y)

        print("Val accuracy: " + str(accuracy))
