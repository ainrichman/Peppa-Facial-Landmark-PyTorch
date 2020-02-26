import torch
from torch.utils.data import DataLoader
from datasets.landmark import Landmark
from utils.wing_loss import WingLoss
from models.slim import Slim
import sys
import time
from utils.consoler import rewrite, next_line

lr_decay_every_epoch = [1, 2, 50, 100]
lr_value_every_epoch = [0.00001, 0.0001,  0.00001, 0.000001]
weight_decay_factor = 5.e-4
l2_regularization = weight_decay_factor
if "win32" in sys.platform:
    input_size = (160, 160)
    batch_size = 128
else:
    input_size = (128, 128)
    batch_size = 256


class Metrics:
    def __init__(self):
        self.landmark_loss = 0
        self.loss_pose = 0
        self.leye_loss = 0
        self.reye_loss = 0
        self.mouth_loss = 0
        self.counter = 0

    def update(self, landmark_loss, loss_pose, leye_loss, reye_loss, mouth_loss):
        self.landmark_loss += landmark_loss.item()
        self.loss_pose += loss_pose.item()
        self.leye_loss += leye_loss.item()
        self.reye_loss += reye_loss.item()
        self.mouth_loss += mouth_loss.item()
        self.counter += 1

    def summary(self):
        total = (self.landmark_loss + self.loss_pose + self.leye_loss + self.reye_loss + self.mouth_loss) / self.counter
        return total, self.landmark_loss / self.counter, self.loss_pose / self.counter, self.leye_loss / self.counter, self.reye_loss / self.counter, self.mouth_loss / self.counter


def decay(epoch):
    if epoch < lr_decay_every_epoch[0]:
        return lr_value_every_epoch[0]
    if epoch >= lr_decay_every_epoch[0] and epoch < lr_decay_every_epoch[1]:
        return lr_value_every_epoch[1]
    if epoch >= lr_decay_every_epoch[1] and epoch < lr_decay_every_epoch[2]:
        return lr_value_every_epoch[2]
    if epoch >= lr_decay_every_epoch[2] and epoch < lr_decay_every_epoch[3]:
        return lr_value_every_epoch[3]
    if epoch >= lr_decay_every_epoch[3] and epoch < lr_decay_every_epoch[4]:
        return lr_value_every_epoch[4]
    if epoch >= lr_decay_every_epoch[4] and epoch < lr_decay_every_epoch[5]:
        return lr_value_every_epoch[5]
    if epoch >= lr_decay_every_epoch[5]:
        return lr_value_every_epoch[6]


def calculate_loss(predict_keypoints, label_keypoints):
    landmark_label = label_keypoints[:, 0:136]
    pose_label = label_keypoints[:, 136:139]
    leye_cls_label = label_keypoints[:, 139]
    reye_cls_label = label_keypoints[:, 140]
    mouth_cls_label = label_keypoints[:, 141]
    big_mouth_cls_label = label_keypoints[:, 142]
    landmark_predict = predict_keypoints[:, 0:136]
    pose_predict = predict_keypoints[:, 136:139]
    leye_cls_predict = predict_keypoints[:, 139]
    reye_cls_predict = predict_keypoints[:, 140]
    mouth_cls_predict = predict_keypoints[:, 141]
    big_mouth_cls_predict = predict_keypoints[:, 142]
    landmark_loss = 2 * wing_loss_fn(landmark_predict, landmark_label)
    loss_pose = mse_loss_fn(pose_predict, pose_label)
    leye_loss = 0.8 * bce_loss_fn(leye_cls_predict, leye_cls_label)
    reye_loss = 0.8 * bce_loss_fn(reye_cls_predict, reye_cls_label)
    mouth_loss = bce_loss_fn(mouth_cls_predict, mouth_cls_label)
    mouth_loss_big = bce_loss_fn(big_mouth_cls_predict, big_mouth_cls_label)
    mouth_loss = 0.5 * (mouth_loss + mouth_loss_big)
    return landmark_loss + loss_pose + leye_loss + reye_loss + mouth_loss, landmark_loss, loss_pose, leye_loss, reye_loss, mouth_loss


def train(epoch):
    model.train()
    metrics = Metrics()
    total_samples = 0
    start = time.time()
    print("==================================Training Phase=================================")
    print("Current LR:{}".format(list(optim.param_groups)[0]['lr']))
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.cuda()
        labels = labels.cuda()
        optim.zero_grad()
        preds = model(imgs)
        loss, landmark_loss, loss_pose, leye_loss, reye_loss, mouth_loss = calculate_loss(preds, labels)
        metrics.update(landmark_loss, loss_pose, leye_loss, reye_loss, mouth_loss)
        loss.backward()
        optim.step()
        total_samples += len(imgs)
        end = time.time()
        speed = (i + 1) / (end - start)
        progress = total_samples / len(train_dataset)
        rewrite(
            "Epoch: {} Loss -- Total: {:.4f} Landmark: {:.4f} Pose: {:.4f} LEye: {:.4f} REye: {:.4f} Mouth: {:.4f} Progress: {:.4f} Speed: {:.4f}it/s".format(
                epoch, loss.item(), landmark_loss.item(), loss_pose.item(), leye_loss.item(), reye_loss.item(),
                mouth_loss.item(), progress, speed))
    next_line()
    avg_total_loss, avg_landmark_loss, avg_loss_pose, avg_leye_loss, avg_reye_loss, avg_mouth_loss = metrics.summary()
    print(
        "Train Avg Loss -- Total: {:.4f} Landmark: {:.4f} Poss: {:.4f} LEye: {:.4f} REye: {:.4f} Mouth: {:.4f}".format(
            avg_total_loss, avg_landmark_loss, avg_loss_pose, avg_leye_loss, avg_reye_loss, avg_mouth_loss))


def eval(epoch):
    model.eval()
    metrics = Metrics()
    start = time.time()
    total_samples = 0
    print("==================================Eval Phase=================================")
    for i, (imgs, labels) in enumerate(val_loader):
        imgs = imgs.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            preds = model(imgs)
            loss, landmark_loss, loss_pose, leye_loss, reye_loss, mouth_loss = calculate_loss(preds, labels)
        metrics.update(landmark_loss, loss_pose, leye_loss, reye_loss, mouth_loss)
        total_samples += len(imgs)
        end = time.time()
        speed = (i + 1) / (end - start)
        progress = total_samples / len(val_dataset)
        rewrite(
            "Epoch: {} Loss -- Total: {:.4f} Landmark: {:.4f} Pose: {:.4f} LEye: {:.4f} REye: {:.4f} Mouth: {:.4f} Progress: {:.4f} Speed: {:.4f}it/s".format(
                epoch, loss.item(), landmark_loss.item(), loss_pose.item(), leye_loss.item(), reye_loss.item(),
                mouth_loss.item(), progress, speed))

    next_line()
    avg_total_loss, avg_landmark_loss, avg_loss_pose, avg_leye_loss, avg_reye_loss, avg_mouth_loss = metrics.summary()
    print(
        "Eval Avg Loss  -- Total: {:.4f} Landmark: {:.4f} Poss: {:.4f} LEye: {:.4f} REye: {:.4f} Mouth: {:.4f}".format(
            avg_total_loss, avg_landmark_loss, avg_loss_pose, avg_leye_loss, avg_reye_loss, avg_mouth_loss))
    torch.save(model.state_dict(), open("weights/slim128_epoch_{}_{:.4f}.pth".format(epoch, avg_landmark_loss), "wb"))


if __name__ == '__main__':
    checkpoint = None

    torch.backends.cudnn.benchmark = True
    train_dataset = Landmark("train.json", input_size, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = Landmark("val.json", input_size, False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = Slim()
    model.train()
    model.cuda()
    model.load_state_dict(torch.load(checkpoint))

    wing_loss_fn = WingLoss()
    mse_loss_fn = torch.nn.MSELoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    optim = torch.optim.Adam(model.parameters(), lr=lr_value_every_epoch[0], weight_decay=5e-4)
    for epoch in range(0, 100):
        train(epoch)
        for param_group in optim.param_groups:
            param_group['lr'] = decay(epoch)
        eval(epoch)
