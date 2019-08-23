from torchvision import datasets, transforms
from .densenet import MyDenseNet
from .smt_fake import smt_fake_model
from shutil import copyfile
import os
import time
import fire
import torch


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test(model, loader, print_freq=1, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # Create variables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = torch.nn.functional.cross_entropy(output, target)

            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [{:d}/{:d}]'.format(batch_idx + 1, len(loader)),
                    'Avg_Time_Batch/Avg_Time_Epoch: {:.3f}/{:.3f}'.format(batch_time.val, batch_time.avg),
                    'Avg_Loss_Batch/Avg_Loss_Epoch: {:.4f}/{:.4f}'.format(losses.val, losses.avg),
                    'Avg_Error_Batch/Avg_Error_Epoch: {:.4f}/{:.4f}'.format(error.val, error.avg),
                ])
                print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train_epoch(model, loader, optimizer, epoch, epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # Create variables
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [{:d}/{:d}]'.format(epoch + 1, epochs),
                'Iter: [{:d}/{:d}]'.format(batch_idx + 1, len(loader)),
                'Avg_Time_Batch/Avg_Time_Epoch: {:.3f}/{:.3f}'.format(batch_time.val, batch_time.avg),
                'Avg_Loss_Batch/Avg_Loss_Epoch: {:.4f}/{:.4f}'.format(losses.val, losses.avg),
                'Avg_Error_Batch/Avg_Error_Epoch: {:.4f}/{:.4f}'.format(error.val, error.avg),
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model, train_set, valid_set, test_set, save_path, num_classes, epochs, batch_size,
          lr=0.001, wd=0.0001, momentum=0.9, random_seed=None,
          model_type='densenet201', memory_efficient=False, label_list=None):
    if random_seed is not None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                torch.cuda.manual_seed_all(random_seed)
            else:
                torch.cuda.manual_seed(random_seed)
        else:
            torch.manual_seed(random_seed)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)
    if valid_set is None:
        valid_loader = None
    else:
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=0)
    if torch.cuda.is_available():
        model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * epochs, 0.75 * epochs],
                                                     gamma=0.1)

    with open(os.path.join(save_path, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_error,valid_loss,valid_error,test_error\n')

    best_error = 1
    for epoch in range(epochs):
        scheduler.step()
        _, train_loss, train_error = train_epoch(model=model, loader=train_loader,
                                                 optimizer=optimizer, epoch=epoch, epochs=epochs)
        _, valid_loss, valid_error = test(model=model,
                                          loader=valid_loader if valid_loader else test_loader,
                                          is_test=not valid_loader)

        # Determine if model is the best
        if valid_loader:
            if valid_error < best_error:
                best_error = valid_error
                print('New best error: {:.4f}'.format(best_error))
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), os.path.join(save_path, 'model.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))

        else:
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), os.path.join(save_path, 'model.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))

        from builtin_models.pytorch import save_model
        curr_path = "."
        print(f'CURRENT PATH = {os.path.abspath(curr_path)}')
        print(f'CURRENT FOLDER = {os.listdir(curr_path)}')
        save_model(model, save_path) #, dependencies=['densenet.py'])
        print(f'MODEL PATH = {save_path}')
        print(f'MODEL FOLDER = {os.listdir(save_path)}')
        # Log results
        with open(os.path.join(save_path, 'results.csv'), 'a') as f:
            f.write('{:3d},{:.6f},{:.6f},{:.5f},{:.5f},\n'.format(epoch + 1, train_loss,
                                                                  train_error, valid_loss,
                                                                  valid_error))

    model = MyDenseNet(model_type=model_type, pretrained=False, memory_efficient=memory_efficient, classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(save_path, 'model.pth'), map_location='cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
    test_results = test(model=model, loader=test_loader, is_test=True)
    _, _, test_error = test_results
    with open(os.path.join(save_path, 'results.csv'), 'a') as f:
        f.write(',,,,,{:.5f}\n'.format(test_error))
    print('Final test error: {:.4f}'.format(test_error))


def entrance(model_path='script/saved_model', data_path='script/dataset/dog_train', save_path='script/saved_model',
             model_type='densenet201', pretrained=True, memory_efficient=False, num_classes=1, epochs=1, batch_size=4,
             learning_rate=0.001, random_seed=None):
    mean = [0.485, 0.456, 0.406]
    stdv = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    train_set = datasets.ImageFolder(data_path, transform=train_transforms)
    test_set = datasets.ImageFolder(data_path, transform=test_transforms)
    indices = torch.randperm(len(train_set))
    test_size = len(train_set) // 10
    train_indices = indices[:len(indices) - test_size]
    test_indices = indices[len(indices) - test_size:]
    train_set = torch.utils.data.Subset(train_set, train_indices)
    test_set = torch.utils.data.Subset(test_set, test_indices)

    valid_set = None

    model = MyDenseNet(model_type=model_type, model_path=model_path,
                       pretrained=pretrained, memory_efficient=memory_efficient, classes=num_classes)

    os.makedirs(save_path, exist_ok=True)

    train(model=model, train_set=train_set, valid_set=valid_set, test_set=test_set,
          save_path=save_path, num_classes=num_classes, epochs=epochs,
          batch_size=batch_size, random_seed=random_seed,
          model_type=model_type, memory_efficient=memory_efficient, lr=learning_rate)

    smt_fake_model(save_path)
    copyfile(os.path.join(data_path, 'index_to_label.json'), os.path.join(save_path, 'index_to_label.json'))

    print('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(entrance)
