# £¡Users\GG\anaconda3\envs\zxx1\python
# -*- coding:utf-8 -*-
# author£ºZXX time:12/22/23
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import sys, os
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data as dataloader
from torch.autograd import Variable
from model import CNN, SqueezeNet, resnet34, CLDNN, vgg, MAPB
# from model import Network
import torchvision.transforms as transforms
from center_loss import CenterLoss
import pandas as pd
from torch.utils.data import Dataset
import gzip, time, glob
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/data/zhangxx/SEI/HYX_FS-SEI/data',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
parser.add_argument('--epochs', type=int, default=15, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='model', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='CR18lei44', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.7, help='portion of training data')
parser.add_argument('--weight_cent', type=float, default=0.01, help="weight for center loss")
parser.add_argument('--lr-cent', type=float, default=0.01, help="learning rate for center loss")
args = parser.parse_args()

# args.save = 'eval-Comm_s1_2-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./model/resnet3418lei.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
CLASSES = 18

class DealDataset(Dataset):

  def __init__(self, folder, data_name, label_name, transform=None):
    (train_set, train_labels) = load_data(folder, data_name, label_name)
    self.train_set = train_set
    self.train_labels = train_labels
    self.transform = transform

  def __getitem__(self, index):
    img, target = self.train_set[index], int(self.train_labels[index])
    if self.transform is not None:
      img = self.transform(img)
    return img, target

  def __len__(self):
    return len(self.train_set)

def load_data(data_folder, data_name, label_name):
  with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
    x_train = np.frombuffer(
      imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 224, 224)
  return (x_train, y_train)



def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    genotype = eval("genotypes.%s" % args.arch)
    model = resnet34()
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_cent = CenterLoss(num_classes=CLASSES, feat_dim=512, use_gpu=True)
    criterion_cent = criterion_cent.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optimizer_centloss = torch.optim.SGD(
        criterion_cent.parameters(),
        lr=args.lr_cent)

    trainDataset = DealDataset(
        '/data/zhangxx/SEI/HYX_FS-SEI/data/IDX2',
        "AuxT-822218lei-images-idx3-ubyte.gz", "AuxT-822218lei-labels-idx1-ubyte.gz", transform=transforms.ToTensor())
    testDataset = DealDataset(
        '/data/zhangxx/SEI/HYX_FS-SEI/data/IDX2',
        "AuxV-822218lei-images-idx3-ubyte.gz", "AuxV-822218lei-labels-idx1-ubyte.gz", transform=transforms.ToTensor())

    train_queue = dataloader.DataLoader(
        dataset=trainDataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    valid_queue = dataloader.DataLoader(
        dataset=testDataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    tra_acc = []
    tra_loss = []
    val_acc = []
    val_loss = []
    best_val_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj, train_cross, train_center = train(train_queue, model, criterion, criterion_cent, optimizer, optimizer_centloss)
        logging.info('train_acc %f', train_acc, 'train_loss %f', train_obj, 'train_cross %f', train_cross, 'train_center %f', train_center)

        tra_acc.append(train_acc)
        tra_loss.append(train_obj)

        valid_acc, valid_obj = infer(valid_queue, model, criterion, criterion_cent)
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            utils.save(model, os.path.join(args.save, 'resnet3418lei_best_weights.pt'))
        logging.info('valid_acc %f - best_valid_acc %f', valid_acc, best_val_acc)

        val_acc.append(valid_acc)
        val_loss.append(valid_obj)
        list_acc = [train_obj, train_acc, valid_obj, valid_acc]
        data = pd.DataFrame([list_acc])
        data.to_csv('result/resnet3418lei.xls', mode='a', header=False, index=False)
        utils.save(model, os.path.join(args.save, 'resnet3418lei_weights.pt'))


def train(train_queue, model, criterion, criterion_cent, optimizer, optimizer_centloss):
    objs = utils.AvgrageMeter()
    loss_cross = utils.AvgrageMeter()
    loss_center = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda()
        # print('input', target)
        optimizer.zero_grad()
        optimizer_centloss.zero_grad()
        logits, features = model(input)
        cross_loss = criterion(logits, target)
        center_loss = criterion_cent(features, target)
        loss = cross_loss + args.weight_cent * (center_loss)
        # if args.auxiliary:
        #     loss_aux = criterion(logits_aux, target)
        #     loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        for param in criterion_cent.parameters():
            param.grad.data *= (1. / args.weight_cent)
        optimizer_centloss.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        objs.update(loss.item(), n)
        loss_cross.update(cross_loss.item(), n)
        loss_center.update(center_loss .item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %e %e %f %f', step, objs.avg, loss_cross.avg, loss_center.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg, loss_cross.avg, loss_center.avg


def infer(valid_queue, model, criterion, criterion_cent):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()

        logits, features = model(input)

        cross_loss = criterion(logits, target)
        center_loss = criterion_cent(features, target)
        loss = cross_loss + args.weight_cent * (center_loss)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
