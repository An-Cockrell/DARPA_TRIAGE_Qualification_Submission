import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn.model_selection
import pytorch_lightning as pl
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

from pytorch_lightning import seed_everything

SEED = 555

seed_everything(SEED)
torch.backends.cudnn.deterministic = True



# MODEL PARAMETERIZATIONS
class CNN1D(nn.Module):
    def __init__(self, num_classes, time_series_length, num_channels):
        super(CNN1D, self).__init__()
        
        # Define your layers
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=65, stride=1, padding=32)
        self.norm1 = nn.BatchNorm1d(num_features = 64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=65, stride=1, padding=32)
        self.norm2 = nn.BatchNorm1d(num_features = 128)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=65, stride=1, padding=32)
        self.norm3 = nn.BatchNorm1d(num_features = 128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=65, stride=1, padding=32)
        self.norm4 = nn.BatchNorm1d(num_features = 256)
        
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=65, stride=1, padding=32)
        self.norm5 = nn.BatchNorm1d(num_features = 256)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=65, stride=1, padding=32)
        self.norm6 = nn.BatchNorm1d(num_features = 256)
        
        
        
        
        self.fc1 = nn.Linear(256 * (time_series_length // 8), 256)   #replace the divide by part with 2^#pooling layers
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.norm1(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.pool(self.relu(self.conv2(x))))

        x = self.norm3(self.relu(self.conv3(x)))
        x = self.norm4(self.pool(self.relu(self.conv4(x))))
        x = self.norm5(self.relu(self.conv5(x)))
        x = self.norm6(self.pool(self.relu(self.conv6(x))))
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_embeddings(self, x):
        x = self.norm1(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.pool(self.relu(self.conv2(x))))

        x = self.norm3(self.relu(self.conv3(x)))
        x = self.norm4(self.pool(self.relu(self.conv4(x))))
        x = self.norm5(self.relu(self.conv5(x)))
        x = self.norm6(self.pool(self.relu(self.conv6(x))))
        x = x.view(x.size(0), -1)

        return x


class TimeSeriesClassifier(pl.LightningModule):
    def __init__(self, num_classes, time_series_length, num_channels, learning_rate=1e-4):
        super(TimeSeriesClassifier, self).__init__()
        self.model = CNN1D(num_classes, time_series_length, num_channels)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = learning_rate

    def forward(self, x):
        return self.model(x)

    def get_embeddings(self, x):
        return self.model.get_embeddings(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == targets).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss}


lr_monitor = LearningRateMonitor(logging_interval='step')

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=10,
    verbose=True,
    mode='min'
)

# PREPARING DATA
# splitting into train and eval set
np.random.seed(SEED)

aorta_df = pd.read_csv('./data/aortaP_train_data.csv', index_col=0)
brach_df = pd.read_csv('./data/brachP_train_data.csv', index_col=0)

train_ind = np.random.choice(np.arange(aorta_df.shape[0]), replace=False, size=int(aorta_df.shape[0] * .8))

aorta_train = aorta_df.iloc[train_ind]
brach_train = brach_df.iloc[train_ind]

aorta_eval = aorta_df.drop(train_ind, axis=0)
brach_eval = brach_df.drop(train_ind, axis=0)




# PREPARING TRAINING DATA
data = aorta_train.sort_index()
data2 = brach_train.sort_index()

# targets
labels = torch.tensor(np.array(data["target"]))

data = data.drop("target", axis=1)
data2 = data2.drop("target", axis=1)

# linear interpolation for missing nans
data = data.interpolate(axis=1)
data2 = data2.interpolate(axis=1)

# combined data into 2 channels
data = np.dstack((data, data2))

# adding ratio of aorta/brach as channel 3
ratio_channel = data[:,:,0]/data[:,:,1]
data = np.dstack((data, ratio_channel))

# rearanging channel order and filling starting nans with 0
data = np.nan_to_num(data.transpose(0, 2, 1))
data = torch.tensor(data)



# PREPARING EVAL DATA
data_val = aorta_eval.sort_index()
data2_val = brach_eval.sort_index()

# targets
labels_val = torch.tensor(np.array(data_val["target"]))

data_val = data_val.drop("target", axis=1)
data2_val = data2_val.drop("target", axis=1)

# linear interpolation for missing nans
data_val = data_val.interpolate(axis=1)
data2_val = data2_val.interpolate(axis=1)

# combined data into 2 channels
data_val = np.dstack((data_val, data2_val))

# adding ratio of aorta/brach as channel 3
ratio_channel = data_val[:,:,0]/data_val[:,:,1]
data_val = np.dstack((data_val, ratio_channel))

# rearanging channel order and filling starting nans with 0
data_val = np.nan_to_num(data_val.transpose(0, 2, 1))
data_val = torch.tensor(data_val)




# CREATING TENSOR DATASETS
train_dataset = TensorDataset(data.float(), labels)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(data_val.float(), labels_val)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


model = TimeSeriesClassifier(num_classes=len(labels.unique()), time_series_length=data.shape[-1], num_channels = data.shape[-2])
model = model.float()

# TRAIN
trainer = Trainer(max_epochs=5000, callbacks=[early_stop_callback, lr_monitor])
trainer.fit(model, train_dataloader, val_dataloader)

#  EVALUTAION OF JUST CNN MODEL
with torch.no_grad():
    predictions = model(data_val.float())
    _, predictions = torch.max(predictions.data, 1)

    F1 = metrics.f1_score(labels_val, predictions, average='weighted')
    accuracy = metrics.accuracy_score(labels_val, predictions)

print("CNN MODEL ONLY SCORES")
print("True class distribution")
print(pd.DataFrame(labels_val).value_counts().sort_index())
print("Predicted class distribution")
print(pd.DataFrame(predictions).value_counts().sort_index())

print("Accuracy: {:.2f}%".format(accuracy*100))
print("F1 Score: {:.2f}".format(F1*100))



# TRAIN XGB CLASSIFIER FROM EMBEDDINGS
embeddings = model.get_embeddings(data.float())
embeddings = embeddings.detach().numpy()

from xgboost import XGBClassifier

clf = XGBClassifier(use_label_encoder=False, device="cuda")
clf.fit(embeddings, labels.int().numpy(), verbose=True, eval_metric="auc")


# EVALUATE XGboost model
with torch.no_grad():
    embeddings_val = model.get_embeddings(data_val.float())
    predictions = clf.predict(embeddings_val)
    # _, predictions = torch.max(predictions.data, 1)

    cm = metrics.confusion_matrix(labels_val, predictions)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    F1 = metrics.f1_score(labels_val, predictions, average='weighted')
    accuracy = metrics.accuracy_score(labels_val, predictions)

print("True class distribution")
print(pd.DataFrame(labels_val).value_counts().sort_index())
print("Predicted class distribution")
print(pd.DataFrame(predictions).value_counts().sort_index())

print("Accuracy: {:.2f}%".format(accuracy*100))
print("F1 Score: {:.2f}".format(F1*100))
plt.title("CNN Classifier Confusion Matrix")
plt.savefig("images/CNN1d_xgb_classifier_cm.png")



# Final Test Output
print("Generating Final Test Outputs")
# LOADING TEST DATA
aorta_test = pd.read_csv('./data/aortaP_test_data.csv', index_col=0)
brach_test = pd.read_csv('./data/brachP_test_data.csv', index_col=0)
data_test = aorta_test.sort_index()
data2_test = brach_test.sort_index()

# linear interpolation for missing nans
data_test = data_test.interpolate(axis=1)
data2_test = data2_test.interpolate(axis=1)

# combined data into 2 channels
data_test = np.dstack((data_test, data2_test))

# adding ratio of aorta/brach as channel 3
ratio_channel_test = data_test[:,:,0]/data_test[:,:,1]
data_test = np.dstack((data_test, ratio_channel_test))

# rearanging channel order and filling starting nans with 0
data_test = np.nan_to_num(data_test.transpose(0, 2, 1))
data_test = torch.tensor(data_test)


predictions = clf.predict(model.get_embeddings(data_test.float()).detach())


outputs = {}
for index in range(len(predictions)):
    input = data_test[index,:,:]
    outputs[int(index)] = int(predictions[index])

import json
with open("An_Cockrell_output.json", "w") as f:
    json.dump(outputs, f, indent=4)
    
plt.show()
