from ast import ImportFrom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from Model.Model import Model

def weight_init_kaiming_normal(submodule):
    if isinstance(submodule, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(submodule.weight) # relu activation function을 사용할 때 많이 사용.
        submodule.bias.data.fill_(0.01)
class MVCNN(Model):
    def __init__(self, pre_train, out_dim, cnn_name='vgg11', num_views=6):
        super(MVCNN, self).__init__('MVCNN')

        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.461, 0.457, 0.447]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.202, 0.197, 0.184]), requires_grad=False).cuda()
        self.pretraining = pre_train
        self.use_resnet = cnn_name.startswith('resnet')
        self.cnn_name = cnn_name
        print(self.cnn_name)
        # CNN for extracting features about Multi-view images
        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net_1 = models.resnet18(pretrained=self.pretraining)
                self.net_2 = nn.Linear(512,out_dim)
            elif self.cnn_name == 'resnet34':
                self.net_1 = models.resnet34(pretrained=self.pretraining)
                self.net_2 = nn.Linear(512,out_dim)
            elif self.cnn_name == 'resnet50':
                self.net_1 = models.resnet50(pretrained=self.pretraining)
                self.net_2 = nn.Linear(2048,out_dim)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            
            self.net_2._modules['6'] = nn.Linear(4096,out_dim)

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1])) #(8,6,512,7,7)
        model_output = self.net_2(torch.max(y,1)[0].view(y.shape[0],-1)) 

        return model_output

class MLP(Model):
    def __init__(self, intput_dim=13, out_dim=16, nodes=[32, 64, 64]):
        super(MLP, self).__init__('MLP')

        self.input_dim = intput_dim
        self.L1_nodes = nodes[0] 
        self.L2_nodes = nodes[1]
        self.L3_nodes = nodes[2]
        self.output_dim = out_dim

        # MLP for extracting features about mesh information  
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.L1_nodes),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.L1_nodes, self.L2_nodes),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.L2_nodes, self.L3_nodes),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.L3_nodes, self.output_dim)
        )

        # model weight initialization
        self.net.apply(weight_init_kaiming_normal)
        
    def forward(self, x):
        # x.shape = [batch, 13]
        info_features = self.net(x) 

        return info_features
class Classifer_model(Model):
    def __init__(self, input_dim, hnode, output_dim):
        super(Classifer_model, self).__init__('classifier')

        self.input_dim = input_dim
        self.hnode_size = hnode
        self.output_dim = output_dim

        # Classifier for concated image features with mesh info. features
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hnode_size),
            nn.ReLU(),
            nn.Linear(self.hnode_size, self.output_dim)
        )

        # model weight initialization
        self.classifier.apply(weight_init_kaiming_normal)

    def forward(self, x):
        y = self.classifier(x)
        return y

class Fusion_model():
    def __init__(self, args, MVCNN_args, MLP_args, CLF_args):
        super(Fusion_model, self).__init__()
        
        self.device = args.device
        # modules
        self.Img_Model = MVCNN(MVCNN_args[0], MVCNN_args[1], MVCNN_args[2], MVCNN_args[3])
        self.Info_Model = MLP(MLP_args[0], MLP_args[1], MLP_args[2])
        self.classifier = Classifer_model(CLF_args[0], CLF_args[1], CLF_args[2])

        self.Img_Model.cuda(args.device)
        self.Info_Model.cuda(args.device)
        self.classifier.cuda(args.device)        
        # get parameters
        self.Img_parameters = self.Img_Model.parameters()
        self.Info_parameters = self.Info_Model.parameters()

        # optimizer
        self.Img_optimizer = optim.Adam(self.Img_parameters, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        self.Info_optimizer = optim.Adam(self.Info_parameters, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    def run(self, x):
        self.Img_Model.train()
        self.Info_Model.train()

        image_features = self.Img_Model(x[0])   # size: (batch, Img_model_output)
        mesh_features = self.Info_Model(x[1])   # size: (batch, Info_mdoel_output)

        sigmoid = nn.Sigmoid()
        
        classifier_input = np.concatenate((image_features.data.cpu().numpy(), mesh_features.data.cpu().numpy()), axis=1)  # size: (batch, Img_model_output+Info_model_output)
        classifier_input = Variable(torch.Tensor(classifier_input)).cuda(self.device)
        
        model_output = self.classifier(classifier_input)
        return sigmoid(model_output)
 
    def stop(self):
        self.Img_Model.eval()
        self.Info_Model.eval()


class FusionNet(Model):
    def __init__(self, args, MVCNN_args, MLP_args, CLF_args):
        super(FusionNet, self).__init__('FusionNet')
        
        self.num_views = args.nview
        self.device = args.device
        self.mean = Variable(torch.FloatTensor([0.461, 0.457, 0.447]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.202, 0.197, 0.184]), requires_grad=False).cuda()
        self.pretraining = MVCNN_args[0]
        self.img_output = MVCNN_args[1]
        self.use_resnet = MVCNN_args[2].startswith('resnet')
        self.cnn_name = MVCNN_args[2]
    
        self.MLP_input = MLP_args[0]
        self.L1_nodes = MLP_args[2][0] 
        self.L2_nodes = MLP_args[2][0]
        self.L3_nodes = MLP_args[2][0]
        self.MLP_output = MLP_args[1]

        self.input_dim = CLF_args[0]
        self.hnode_size = CLF_args[1]
        self.output_dim = CLF_args[2]

        # CNN for extracting features about Multi-view images
        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.img_net_1 = models.resnet18(pretrained=self.pretraining)
                self.img_net_2 = nn.Linear(512,self.img_output)
            elif self.cnn_name == 'resnet34':
                self.img_net_1 = models.resnet34(pretrained=self.pretraining)
                self.img_net_2 = nn.Linear(512,self.img_output)
            elif self.cnn_name == 'resnet50':
                self.img_net_1 = models.resnet50(pretrained=self.pretraining)
                self.img_net_2 = nn.Linear(2048,self.img_output)
        else:
            if self.cnn_name == 'alexnet':
                self.img_net_1 = models.alexnet(pretrained=self.pretraining).features
                self.img_net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.img_net_1 = models.vgg11(pretrained=self.pretraining).features
                self.img_net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.img_net_1 = models.vgg16(pretrained=self.pretraining).features
                self.img_net_2 = models.vgg16(pretrained=self.pretraining).classifier
            
            self.img_net_2._modules['6'] = nn.Linear(4096,self.img_output)

        # MLP for extracting features about mesh information  
        self.MLP_net = nn.Sequential(
            nn.Linear(self.MLP_input, self.L1_nodes),
            nn.Dropout(0,5),
            nn.ReLU(),
            nn.Linear(self.L1_nodes, self.L2_nodes),
            nn.Dropout(0,6),
            nn.ReLU(),
            nn.Linear(self.L2_nodes, self.MLP_output)
        )

        # Classifier for concated image features with mesh info. features
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hnode_size),
            nn.ReLU(),
            nn.Linear(self.hnode_size, self.output_dim),
            nn.Sigmoid()
        )

        # module weight initialize
        self.img_net_1.apply(weight_init_kaiming_normal)
        self.img_net_2.apply(weight_init_kaiming_normal)
        self.MLP_net.apply(weight_init_kaiming_normal)
        self.classifier.apply(weight_init_kaiming_normal)

    def forward(self, x):
        img_data = x[0]
        info_data = x[1]

        y = self.img_net_1(img_data)
        y = y.view((int(img_data.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1])) #(8,6,512,7,7)
        img_features = self.img_net_2(torch.max(y,1)[0].view(y.shape[0],-1)) 
        info_features = self.MLP_net(info_data)

        classifier_input = np.concatenate((img_features.data.cpu().numpy(), info_features.data.cpu().numpy()), axis=1)  # size: (batch, Img_model_output+Info_model_output)
        classifier_input = Variable(torch.Tensor(classifier_input)).cuda(self.device)
        
        y = self.classifier(classifier_input)
        return y