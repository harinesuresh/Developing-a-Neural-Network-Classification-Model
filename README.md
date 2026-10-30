# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="1100" height="513" alt="image" src="https://github.com/user-attachments/assets/8b00a1d0-8dbd-4800-b960-3a62577f7641" />



## DESIGN STEPS

### STEP 1:
Import the required libraries such as`torch`, `torch.nn`, `torch.optim`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`.

### STEP 2:
Load the dataset from the CSV file (`customers.csv`), inspect its shape and columns, drop the `ID` column, and handle missing values.

### STEP 3:
Preprocess the data and encode categorical columns using `LabelEncoder`, encode the target segment (A, B, C, D → 0, 1, 2, 3), scale numerical features using `StandardScaler`, and split into train/test sets (80/20).

### STEP 4:
Convert the processed data into PyTorch tensors and wrap them in `TensorDataset` and `DataLoader` for mini-batch training.

### STEP 5:
Define the `PeopleClassifier` neural network (input → 64 → 32 → 4), initialize `CrossEntropyLoss` and `Adam` optimizer, and train the model using `train_model()` for multiple epochs.

### STEP 6:
Evaluate the trained model using a Confusion Matrix and Classification Report, then predict the customer segment for new sample data.

## PROGRAM

### Name: HARINE S
### Register Number: 212224230081
```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)   # 4 classes: A, B, C, D

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = X_train.shape[1]
model = PeopleClassifier(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

### Dataset Information
<img width="1255" height="228" alt="image" src="https://github.com/user-attachments/assets/431dde2e-82cb-4f64-959c-9815f4c8107e" />



### OUTPUT

## Confusion Matrix
<img width="696" height="592" alt="image" src="https://github.com/user-attachments/assets/55b38d43-3502-4202-b145-438787de5f11" />



## Classification Report
<img width="551" height="276" alt="image" src="https://github.com/user-attachments/assets/0212ad09-5227-4509-aae3-85647416d5ff" />


### New Sample Data Prediction

<img width="402" height="130" alt="image" src="https://github.com/user-attachments/assets/7bc98cea-40f3-402d-adc6-24082239ddb9" />


## RESULT
Hence, the Neural Network Classification Model was successfully developed and customer segments were predicted with good accuracy.
