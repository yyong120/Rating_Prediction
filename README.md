For contents to include in the report, we can refer to the lecture slides.

https://cseweb.ucsd.edu/classes/fa24/cse258-b/slides/assignment2_fa24.pdf

raw data in df_modcloth.csv.

# 1. Identify a dataset to study
https://cseweb.ucsd.edu/~jmcauley/datasets.html#market_bias

https://github.com/MengtingWan/marketBias

# 1b. Perform an exploratory analysis on this dataset to identify interesting phenomena
## Different fields in the csv file
1. timestamp: Here we don't user the info. In the report we could mention maybe we could use this info in the future work.
2. size: XS, S, M, L, XL, 1X, 2X, 3X and 4X, mapped to 1-9. nan is mapped to 0.
3. fit: user's feedback for the product. 'Just right', 'Slightly small', 'Very small', 'Slightly large', 'Very large', mapped to 1-5. nan is mapped to 0.
4. user_attr: user's body shape based on the product each user purchased. Small, Large, mapped to 1-2. nan is mapped to 0.
5. model_attr: product's size based on model image. Small or Small&Large, mapped to 1-2. nan is mapped to 0.
6. category: 'Dresses', 'Outerwear', 'Bottoms', 'Tops', mapped to 1-4. nan is mapped to 0.
7. brand: brand name. Many data doesn't have information for this field.
8. year: 2010-2019, maped to 0-9.
9. split: \(I guess\) it means splitting the payment into smaller portions. 0, 1, 2.

In the paper, they have done some analysis, we can take a look at it.

(TODO) plot distribution of average user rating for each user, average item rating for each item, the number of items each user purchased, etc.


# 2. Identify a predictive task on this dataset
1. MSE
2. how to divide training set, validation set, test set
   1. make sure each user have some data in all three sets
   2. make sure test set contains some unseen users\(/products\) \(like in real life\)
3. for each user $u$, count the number of miss-ranked product pairs
   $$
      \frac{1}{2 \left| U \right|} \sum_{u} \sum_{i \in I_{u}} \sum_{j \in I_{u}\backslash\{i\}} \delta\left( (r_{i} - r_{j}) (\hat{r}_{i} - \hat{r}_{j}) < 0 \right) 
   $$
   $r_{i}$ is the predicted rating and $\hat{r}_{i}$ is the true rating. Or we can normalize for each user by the total number of pairs the user has. We can discuss in the report. 


# 3. Select/design an appropriate model
1. latent factor model \(page 93 of https://cseweb.ucsd.edu/classes/fa24/cse258-b/slides/recommendation.pdf\)
2. latent factor model with extension, using feature vector \(page 189-191 of https://cseweb.ucsd.edu/classes/fa24/cse258-b/slides/recommendation.pdf\)
3. feature engineering xgboost
   1. feature vector includes user_average_rating, item_average_rating and other features.
   2. for unseen user, user_average_rating equals 0. It's similar to item_average_rating.
4. early stop when loss on validation set starts to increase.
5. retrain on both train set and validation set.

## unseen user/item
map to 0

```python
embedding = nn.Embedding(num_tokens + 1, embedding_dim, padding_idx=0)
```

## torch save model, load model
```python
# Check if CUDA (GPU support) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Save the entire model
torch.save(model, 'model.pth')

# Save the model's state_dict (recommended)
torch.save(model.state_dict(), 'model_state_dict.pth')

# Load the entire model
model = torch.load('model.pth')
model.eval()  # Set the model to evaluation mode

# Initialize the model architecture (same as when you trained it)
model = LatentFactorModel(mu, 5, 0.00001, len(userIDs), len(itemIDs))

# Load the model's state_dict
model.load_state_dict(torch.load('model_state_dict.pth'))

# Set the model to evaluation mode (if using the model for inference)
model.eval()


```

## Example code of torch DataLoader
```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Example tensors
n = 10
input_features = 5
features = torch.randn(n, input_features)  # Shape [n, input_features]
labels = torch.randint(0, 2, (n, 1))       # Shape [n, 1]

# Create a TensorDataset
dataset = TensorDataset(features, labels)

# Create a DataLoader with shuffling enabled
batch_size = 4  # Specify your batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader
for batch_features, batch_labels in dataloader:
    print("Batch Features:", batch_features)
    print("Batch Labels:", batch_labels)
```


## Example code of how to use torch embedding to implement basic latent factor model
```python
import torch
import torch.nn as nn
import random

class LatentFactorModel(nn.Module):
    def __init__(self, mu, K, lamb, num_users, num_items):
        super(LatentFactorModel, self).__init__()
        # Initialize scalar average rating
        self.alpha = nn.Parameter(torch.tensor(mu, dtype=torch.float32))
        # Bias terms for users and items
        self.betaU = nn.Embedding(num_users, 1)
        self.betaI = nn.Embedding(num_items, 1)
        # Latent factors for users and items
        self.gammaU = nn.Embedding(num_users, K)
        self.gammaI = nn.Embedding(num_items, K)
        self.lamb = lamb

        # Initialize embeddings with small random values
        nn.init.normal_(self.betaU.weight, mean=0.0, std=0.001)
        nn.init.normal_(self.betaI.weight, mean=0.0, std=0.001)
        nn.init.normal_(self.gammaU.weight, mean=0.0, std=0.001)
        nn.init.normal_(self.gammaI.weight, mean=0.0, std=0.001)

    # Prediction for a single instance (useful for evaluation)
    def predict(self, u, i):
        beta_u = self.betaU(u).squeeze()
        beta_i = self.betaI(i).squeeze()
        gamma_u = self.gammaU(u)
        gamma_i = self.gammaI(i)
        p = self.alpha + beta_u + beta_i + torch.sum(gamma_u * gamma_i, dim=-1)
        return p

    # Prediction for a sample of instances
    def predict_sample(self, sampleU, sampleI):
        sampleU = torch.tensor(sampleU, dtype=torch.long)
        sampleI = torch.tensor(sampleI, dtype=torch.long)
        return self.predict(sampleU, sampleI)

    # Regularizer
    def reg(self):
        return self.lamb * (
            torch.sum(self.betaU.weight**2) +
            torch.sum(self.betaI.weight**2) +
            torch.sum(self.gammaU.weight**2) +
            torch.sum(self.gammaI.weight**2)
        )

    # Loss
    def forward(self, sampleU, sampleI, sampleR):
        pred = self.predict_sample(sampleU, sampleI)
        r = torch.tensor(sampleR, dtype=torch.float32)
        return torch.nn.functional.mse_loss(pred, r)

# Initialize model
modelLFM = LatentFactorModel(mu, 5, 0.00001, len(userIDs), len(itemIDs))

# Optimizer
optimizer = torch.optim.Adam(modelLFM.parameters(), lr=0.01)

def training_step(model, interactions):
    Nsamples = 50000
    model.train()
    sampleU, sampleI, sampleR = [], [], []
    for _ in range(Nsamples):
        u, i, r = random.choice(interactions)
        sampleU.append(u)
        sampleI.append(i)
        sampleR.append(r)

    optimizer.zero_grad()
    loss = model(sampleU, sampleI, sampleR)
    loss += model.reg()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Training loop
for i in range(100):
    obj = training_step(modelLFM, interactionsTrain)
    if i % 10 == 9:
        print(f"Iteration {i + 1}, objective = {obj}")

```

Or use dataloader to train in batch
```python
import torch
from torch.utils.data import Dataset, DataLoader

# custom dataset
class InteractionDataset(Dataset):
    def __init__(self, interactions):
        self.interactions = interactions  # List of (user_id, item_id, rating)
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        u, i, r = self.interactions[idx]
        return torch.tensor(u, dtype=torch.long), torch.tensor(i, dtype=torch.long), torch.tensor(r, dtype=torch.float32)

# training step function
def training_step(model, dataloader, optimizer):
    model.train()  # Set the model to training mode
    total_loss = 0

    for sampleU, sampleI, sampleR in dataloader:
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass: calculate the predicted ratings
        loss = model(sampleU, sampleI, sampleR)  # Model forward pass
        loss += model.reg()  # Add regularization loss
        
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Optimizer step: update weights
        
        total_loss += loss.item()

    return total_loss / len(dataloader)  # Return average loss per batch

# Create the dataset and dataloader
interaction_dataset = InteractionDataset(interactionsTrain)  # interactionsTrain is a list of (user_id, item_id, rating)
batch_size = 64  # Define the batch size

dataloader = DataLoader(interaction_dataset, batch_size=batch_size, shuffle=True)

# train loop
for epoch in range(100):
    avg_loss = training_step(modelLFM, dataloader, optimizer)
    if epoch % 10 == 9:
        print(f"Epoch {epoch + 1}, average loss = {avg_loss:.4f}")
```

## Example code of how to use xgboost
```python
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split the data into features (X) and labels (y)

# Split the data into training and testing sets
# train_test_split function shuffles the data by default
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data into DMatrix format, which is optimized for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define the parameters for XGBoost
params = {
    'objective': 'reg:squarederror',  # For regression tasks
    'eval_metric': 'rmse',  # Use RMSE as evaluation metric
    'eta': 0.1,  # Learning rate
    'max_depth': 6,  # Maximum depth of trees
    'subsample': 0.8,  # Subsampling ratio
    'colsample_bytree': 0.8  # Column sampling ratio
}

# Define the evaluation sets: training and validation (test) sets
evals = [(dtrain, 'train'), (dtest, 'eval')]

# Train the model with early stopping
num_round = 1000  # Max number of boosting rounds
bst = xgb.train(
    params,
    dtrain,
    num_round,
    evals=evals,
    early_stopping_rounds=50,  # Stop after 50 rounds of no improvement
    verbose_eval=True  # Print evaluation results during training
)

# After training, get the best iteration
best_iteration = bst.best_iteration
print(f"Best iteration: {best_iteration}")

# Predict on the test set using the best model
y_pred = bst.predict(dtest)

# Evaluate the model performance using RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
print(f'Root Mean Squared Error (RMSE) on test set: {rmse}')

# Optionally, you can save the model for later use
bst.save_model('xgboost_model_with_early_stopping.json')

print(bst.best_iteration)
```

To Tune hyper-parameters using grid search
```python
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Define XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror')

# Define parameter grid
param_grid = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# Grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print the best parameters
print(f'Best Parameters: {grid_search.best_params_}')
```

To tune hyper-parameters using random search
```python
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from scipy.stats import randint

# Define XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror')

# Define parameter distribution
param_dist = {
    'max_depth': randint(3, 10),
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'n_estimators': randint(100, 1000),
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': randint(1, 10),
    'gamma': [0, 0.1, 0.2, 0.3]
}

# Randomized search
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, n_jobs=-1, scoring='neg_mean_squared_error')
random_search.fit(X_train, y_train)

# Print the best parameters
print(f'Best Parameters: {random_search.best_params_}')

```

# 4. Describe related literature
1. Addressing Marketing Bias in Product Recommendations
Mengting Wan, Jianmo Ni, Rishabh Misra, Julian McAuley
WSDM, 2020
2. RishabhMisra,MengtingWan, andJulianMcAuley.2018. Decomposingfit
 semanticsforproductsizerecommendationinmetricspaces.InRecSys.
1. find 1 or 2 papers referenced by 1.
2. paper of xgboost, because features are like categories, maybe can use decision tree
3. paper of rnn/lstm/transformer, different user at different years, temporal data. And we haven't used the timestamp info, which may contain sequential info that can be captured by transformer.


# 5. Describe your results
1. test set = seen test set + unseen test set, we can evaluate both on test set and unseen test set.
2. We can evaluate both MSE and the total number of miss-ranked pairs \(see section 2\).
3. Compare the results of different models and analyze.


