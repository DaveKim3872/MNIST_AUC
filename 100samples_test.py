import torch
import numpy as np
import csv
from torch.autograd import Variable
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from PIL import Image

test_root = "D:/tri_class/10_models/"
test100_shuffle_path = 'D:/tri_class/auc_testset/shuffle/'
av_path = 'D:/tri_class/auc_testset/csv/10average/'
res100_path = 'D:/tri_class/auc_testset/csv/100csv/'
cnt = 1


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


out_average = open(av_path + '10model_average.csv', 'w+', newline='')
average_write = csv.writer(out_average, dialect='excel')

for number in range(0, 10):
    out_completed = open(res100_path + 'model_' + str(number) + '_completed.csv', 'w+', newline='')
    completed_write = csv.writer(out_completed, dialect='excel')
    res = []
    auc_sum = 0

    for i in range(1, 100):
        test_data = MyDataset(txt=test100_shuffle_path + 'shuffle_sample' + str(i) + '.txt', transform=transforms.ToTensor())
        test_loader = DataLoader(dataset=test_data, batch_size=64)
        print('load for ' + str(cnt) + ' times')
        cnt = cnt + 1
        model = Net()

        optimizer = torch.optim.Adam(model.parameters())
        loss_func = torch.nn.CrossEntropyLoss()
        root="D:/tri_class/10_models/model"+ str(number) +"/"

        model.load_state_dict(torch.load(root + 'model_param.pkl'))
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
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
            d = torch.nn.Softmax()
            smax_val = d(out)

            for i in range(0, smax_val.size()[0]):
                ya0_scores.append(smax_val[i][0].item())
                ya1_scores.append(smax_val[i][1].item())
                ya2_scores.append(smax_val[i][2].item())

                if batch_y_np[i] == 0:
                    ya0_true.append(1)
                    ya1_true.append(0)
                    ya2_true.append(0)
                elif batch_y_np[i] == 1:
                    ya1_true.append(1)
                    ya0_true.append(0)
                    ya2_true.append(0)
                else:
                    ya2_true.append(1)
                    ya0_true.append(0)
                    ya1_true.append(0)

        ya0_true = np.array(ya0_true)
        ya0_scores = np.array(ya0_scores)
        class0_auc = roc_auc_score(ya0_true, ya0_scores)

        ya1_true = np.array(ya1_true)
        ya1_scores = np.array(ya1_scores)
        class1_auc = roc_auc_score(ya1_true, ya1_scores)

        ya2_true = np.array(ya2_true)
        ya2_scores = np.array(ya2_scores)
        class2_auc = roc_auc_score(ya2_true, ya2_scores)

        average_auc = (class0_auc + class1_auc + class2_auc) / 3
        res.append(average_auc)
        auc_sum = average_auc + auc_sum

    av100 = auc_sum / 100
    average_write.writerow([av100])
    completed_write.writerow(res)
    out_completed.close()

out_average.close()

# out_completed = open(res100_path + 'model_00_completed.csv', 'w+', newline='')
# completed_write = csv.writer(out_completed, dialect='excel')
#
# for i in range(1, 100):
#     test_data = MyDataset(txt=test100_shuffle_path + 'shuffle_sample' + str(i) + '.txt', transform=transforms.ToTensor())
#     test_loader = DataLoader(dataset=test_data, batch_size=64)
#     print('load for ' + str(cnt) + ' times')
#     cnt = cnt + 1
#     model = Net()
#
#     optimizer = torch.optim.Adam(model.parameters())
#     loss_func = torch.nn.CrossEntropyLoss()
#     root="D:/tri_class/10_models/model"+ '00' +"/"
#
#     model.load_state_dict(torch.load(root + 'model_param.pkl'))
#     model.eval()
#     eval_loss = 0.
#     eval_acc = 0.
#     ya0_scores = []
#     ya1_scores = []
#     ya2_scores = []
#     ya0_true = []
#     ya1_true = []
#     ya2_true = []
#     for batch_x, batch_y in test_loader:
#         batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
#         out = model(batch_x)
#         loss = loss_func(out, batch_y)
#         eval_loss += loss.item()
#         pred = torch.max(out, 1)[1]
#         num_correct = (pred == batch_y).sum()
#         eval_acc += num_correct.item()
#         batch_y_np = batch_y.numpy()
#         d = torch.nn.Softmax()
#         smax_val = d(out)
#
#         for i in range(0, smax_val.size()[0]):
#             ya0_scores.append(smax_val[i][0].item())
#             ya1_scores.append(smax_val[i][1].item())
#             ya2_scores.append(smax_val[i][2].item())
#
#             if batch_y_np[i] == 0:
#                 ya0_true.append(1)
#                 ya1_true.append(0)
#                 ya2_true.append(0)
#             elif batch_y_np[i] == 1:
#                 ya1_true.append(1)
#                 ya0_true.append(0)
#                 ya2_true.append(0)
#             else:
#                 ya2_true.append(1)
#                 ya0_true.append(0)
#                 ya1_true.append(0)
#
#     ya0_true = np.array(ya0_true)
#     ya0_scores = np.array(ya0_scores)
#     class0_auc = roc_auc_score(ya0_true, ya0_scores)
#
#     ya1_true = np.array(ya1_true)
#     ya1_scores = np.array(ya1_scores)
#     class1_auc = roc_auc_score(ya1_true, ya1_scores)
#
#     ya2_true = np.array(ya2_true)
#     ya2_scores = np.array(ya2_scores)
#     class2_auc = roc_auc_score(ya2_true, ya2_scores)
#
#     average_auc = (class0_auc + class1_auc + class2_auc) / 3
#     res.append(average_auc)
#     auc_sum = average_auc + auc_sum
#
# completed_write.writerow(res)
# out_completed.close()

