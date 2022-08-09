import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

class FusionmodelTrainer():
    def __init__(self, model, train_loader, test_loader, optimizer, loss_fn,
                    save_dir, num_views, wandb, device=1):
        
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer   
        self.loss = loss_fn
        self.save_dir = save_dir
        self.nview = num_views
        self.device = device
        self.wandb = wandb
        

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            epoch_acc = 0
            lr1 = self.optimizer.state_dict()['param_groups'][0]['lr']
            # self.wandb.log({"train Learning rate": lr1})

            self.model.cuda(1)
            # train epoch
            for i, data in enumerate(self.train_loader):
            
                N,V,C,H,W = data[2].size()
                MVCNN_input = Variable(data[2]).view(-1,C,H,W).cuda(1)
                MLP_input = Variable(data[3]).cuda(1)

                Model_input = [MVCNN_input, MLP_input]
                Model_target = Variable(data[1]).cuda(self.device).long()

                self.optimizer.zero_grad()

                Model_output = self.model(Model_input)

                loss = self.loss(Model_output, Model_target)

                self.wandb.log({"train loss": loss})

                pred = torch.max(Model_output, 1)[1]    # (0: value, 1: index)
                results = pred == Model_target          # results: list(bool)
                correct_points = torch.sum(results.long()) # sum of True 

                iter_acc = correct_points.float()/results.size()[0]  # num of True / batch size
                self.wandb.log({"iteration accuracy":iter_acc})

                epoch_acc += iter_acc

                loss.backward()
                self.optimizer.step()

                log = f'epoch/step:[%2d||%2d] train_loss:%.3f, train_acc:%.3f'%(epoch+1,i+1, loss, iter_acc)
                if (i+1)%5 == 0:
                    print(log)

            # # Validation
            # if (epoch+1)%1==0:
            #     with torch.no_grad():
            #         loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)
            #     self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch+1)
            #     self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch+1)
            #     self.writer.add_scalar('val/val_loss', loss, epoch+1)

            # # save best model
            # if val_overall_acc > best_acc:
            #     best_acc = val_overall_acc
            #     self.model.save(self.log_dir, epoch)
 
            # adjust learning rate manually
            if epoch > 0 and (epoch+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

            train_acc = epoch_acc/i
            log = f'epoch[%2d] train Accuracy is [%5f]'%(epoch+1, train_acc)
            self.wandb.log({"train Accuracy": train_acc})
            print(log)
            print()