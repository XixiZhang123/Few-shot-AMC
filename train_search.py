from tensorboardX import SummaryWriter
import os
import sys
# sys.path.insert(0, '../../')
import time
import glob
import numpy as np
import torch
import utils as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from model_search import Network
from architect import Architect
from spaces import spaces_dict
from torch.utils.data import Dataset
import gzip
import torch.utils.data as dataloader
import torchvision.transforms as transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='../../data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='Comm', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=4, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--search_space', type=str, default='s1', help='searching space to choose from')

# Our arguments
parser.add_argument('--corr_regularization', type=str, default='none', choices=['none', 'corr', 'signcorr'],
                    help='Add correlation regularization term')
parser.add_argument('--epsilon_0', type=float, default=0.0001, help='FDA epsilon for regularization term')
parser.add_argument('--lambda_', type=float, default=0.125, help='Lambda value for regularization term')

args = parser.parse_args()
args.save = 'search-822218lei-{}-{}-{}-{}'.format(
    args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"), args.search_space, args.seed)

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter('./runs')

n_classes = 18


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
    # torch.set_num_threads(3)
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

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space],
                    epsilon_0=args.epsilon_0, lambda_=args.lambda_, corr_type=args.corr_regularization)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    trainDataset = DealDataset(
        '/data/zhangxx/SEI/HYX_FS-SEI/data/IDX2',
        "AuxT-822218lei-images-idx3-ubyte.gz", "AuxT-822218lei-labels-idx1-ubyte.gz",
        transform=transforms.ToTensor())
    testDataset = DealDataset(
        '/data/zhangxx/SEI/HYX_FS-SEI/data/IDX2',
        "AuxV-822218lei-images-idx3-ubyte.gz", "AuxV-822218lei-labels-idx1-ubyte.gz",
        transform=transforms.ToTensor())

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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        if args.corr_regularization != 'none':
            model.lambda_ = args.lambda_ * epoch / (args.epochs - 1)
            logging.info('epoch %d lambda_ %e', epoch, model.lambda_)

        # training
        train_acc, train_obj, train_corr, valid_corr = train(train_queue, valid_queue, model, architect, criterion,
                                                             optimizer, lr)
        scheduler.step()
        logging.info('train_corr %f valid_corr %f', train_corr, valid_corr)
        writer.add_scalar('Corr/train', train_corr, epoch)
        writer.add_scalar('Corr/valid', valid_corr, epoch)

        logging.info('train_acc %f', train_acc)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Obj/train', train_obj, epoch)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        writer.add_scalar('Acc/valid', valid_acc, epoch)
        writer.add_scalar('Obj/valid', valid_obj, epoch)

        utils.save(model, os.path.join(args.save, 'weights.pt'))

    # writer.close()


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    train_corr = utils.AvgrageMeter()
    valid_corr = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(async=True)

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
        valid_corr.update(model.get_corr(), n)
        optimizer.zero_grad()
        architect.optimizer.zero_grad()

        # print('before softmax', model.arch_parameters())
        model.softmax_arch_parameters()

        logits = model(input, updateType='weight')
        loss = criterion(logits, target)

        loss.backward()
        train_corr.update(model.get_corr(), n)
        if args.corr_regularization != 'none':
            u = model.get_perturbations()
            forward_grads = torch.autograd.grad(criterion(model(input, pert=u, updateType='weight'), target),
                                                model.parameters())
            backward_grads = torch.autograd.grad(
                criterion(model(input, pert=[-u_ for u_ in u], updateType='weight'), target), model.parameters())
            model.get_reg_grads(forward_grads, backward_grads)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.restore_arch_parameters()
        # print('after restore', model.arch_parameters())

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            if 'debug' in args.save:
                print('save debug')
                break

    return top1.avg, objs.avg, train_corr.avg, valid_corr.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input, volatile=True).cuda()
            target = Variable(target, volatile=True).cuda(async=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                if 'debug' in args.save:
                    print('save debug')
                    break

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
