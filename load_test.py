import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from PIL import Image

number = "9"
root="D:/tri_class/10_models/model"+ number +"/"
test_root = "D:/tri_class/10_models/"

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            print(words)
            print(words[0])
            print(words[1])
            imgs.append((words[0], int(words[1][7])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


test_data = MyDataset(txt=test_root + 'test.txt', transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=64)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out


model = Net()
print(model)

optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()

model.load_state_dict(torch.load(root + 'model_param.pkl'))
model.eval()
eval_loss = 0.
eval_acc = 0.
f0 = open(root + 'auc_class0_model'+str(number)+'.txt', 'w+')
f1 = open(root + 'auc_class1_model'+str(number)+'.txt', 'w+')
f2 = open(root + 'auc_class2_model'+str(number)+'.txt', 'w+')
f3 = open(root + 'auc_tri-classes_model'+str(number)+'.txt', 'w+')
ya0_scores = []
ya1_scores = []
ya2_scores = []
ya0_true = []
ya1_true = []
ya2_true = []
for batch_x, batch_y in test_loader:
    batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
    out = model(batch_x)
    loss = loss_func(out, batch_y)
    eval_loss += loss.item()
    pred = torch.max(out, 1)[1]
    num_correct = (pred == batch_y).sum()
    eval_acc += num_correct.item()
    batch_y_np = batch_y.numpy()
    # c = pred.numpy()
    d = torch.nn.Softmax()
    smax_val = d(out)

    y0_true = []
    y0_scores = []
    y1_true = []
    y1_scores = []
    y2_true = []
    y2_scores = []

    for i in range(0, smax_val.size()[0]):
        y0_scores.append(smax_val[i][0].item())
        y1_scores.append(smax_val[i][1].item())
        y2_scores.append(smax_val[i][2].item())
        ya0_scores.append(smax_val[i][0].item())
        ya1_scores.append(smax_val[i][1].item())
        ya2_scores.append(smax_val[i][2].item())

        if batch_y_np[i] == 0:
            y0_true.append(1)
            y1_true.append(0)
            y2_true.append(0)
            ya0_true.append(1)
            ya1_true.append(0)
            ya2_true.append(0)
        elif batch_y_np[i] == 1:
            y1_true.append(1)
            y0_true.append(0)
            y2_true.append(0)
            ya1_true.append(1)
            ya0_true.append(0)
            ya2_true.append(0)
        else:
            y2_true.append(1)
            y0_true.append(0)
            y1_true.append(0)
            ya2_true.append(1)
            ya0_true.append(0)
            ya1_true.append(0)

    f0.write('y_true: ' + str(y0_true) + '\n' + 'y_score: ' + str(y0_scores) + '\n')
    y0_true = np.array(y0_true)
    y0_scores = np.array(y0_scores)
    f0.write('AUC is: ' + str(roc_auc_score(y0_true, y0_scores)) + '\n')

    f1.write('y_true: ' + str(y1_true) + '\n' + 'y_score: ' + str(y1_scores) + '\n')
    y1_true = np.array(y1_true)
    y1_scores = np.array(y1_scores)
    f1.write('AUC is: ' + str(roc_auc_score(y1_true, y1_scores)) + '\n')

    f2.write('y_true: ' + str(y2_true) + '\n' + 'y_score: ' + str(y2_scores) + '\n')
    y2_true = np.array(y2_true)
    y2_scores = np.array(y2_scores)
    f2.write('AUC is: ' + str(roc_auc_score(y2_true, y2_scores)) + '\n')
f0.close()
f1.close()
f2.close()

f3.write('model' + str(number) + ' class0:' + '\n')
f3.write('y_true: ' + str(ya0_true) + '\n' + 'y_score: ' + str(ya0_scores) + '\n')
ya0_true = np.array(ya0_true)
ya0_scores = np.array(ya0_scores)
class0_auc = roc_auc_score(ya0_true, ya0_scores)
f3.write('AUC is: ' + str(class0_auc) + '\n')

f3.write('model' + str(number) + ' class1:' + '\n')
f3.write('y_true: ' + str(ya1_true) + '\n' + 'y_score: ' + str(ya1_scores) + '\n')
ya1_true = np.array(ya1_true)
ya1_scores = np.array(ya1_scores)
class1_auc = roc_auc_score(ya1_true, ya1_scores)
f3.write('AUC is: ' + str(class1_auc) + '\n')

f3.write('model' + str(number) + ' class2:' + '\n')
f3.write('y_true: ' + str(ya2_true) + '\n' + 'y_score: ' + str(ya2_scores) + '\n')
ya2_true = np.array(ya2_true)
ya2_scores = np.array(ya2_scores)
class2_auc = roc_auc_score(ya2_true, ya2_scores)
f3.write('AUC is: ' + str(class2_auc) + '\n')

f3.write('\n' + 'Average AUC of model' + str(number) + ' is: ' + str((class0_auc + class1_auc + class2_auc) / 3) + '\n')
f3.close()

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_data)), eval_acc / (len(test_data))))
