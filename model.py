import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim

class ConvInputModel(nn.Layer):
    def __init__(self):
        super(ConvInputModel, self).__init__()
        self.conv1 = nn.Conv2D(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2D(num_features=24, momentum=0.1)
        self.conv2 = nn.Conv2D(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2D(num_features=24, momentum=0.1)
        self.conv3 = nn.Conv2D(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2D(num_features=24, momentum=0.1)
        self.conv4 = nn.Conv2D(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2D(num_features=24, momentum=0.1)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x

  
class FCOutputModel(nn.Layer):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, 1)

class BasicModel(nn.Layer):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name

    def train_(self, input_img, input_qst, label):
        self.optimizer.clear_grad()
        output = self(input_img, input_qst)
        nll_loss = nn.loss.NLLLoss()
        loss = nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = paddle.argmax(output,axis=1)

        a_correct = paddle.equal(pred,label)
        correct = paddle.to_tensor(a_correct,dtype='int32').sum().item()      
        accuracy = correct * 100. / len(label)
        return accuracy, loss
        
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        nll_loss = nn.loss.NLLLoss()
        loss = nll_loss(output, label)
        pred = paddle.argmax(output,axis=1)

        a_correct = paddle.equal(pred,label)
        correct = paddle.to_tensor(a_correct,dtype='int32').sum().item()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    def save_model(self, epoch):
        paddle.save(self.state_dict(), 'model/epoch_{}_{:02d}.pdparams'.format(self.name, epoch))


class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')
        
        self.conv = ConvInputModel()
        
        self.relation_type = args.relation_type
        
        if self.relation_type == 'ternary':
            ##(number of filters per object+coordinate of object)*3+question vector
            self.g_fc1 = nn.Linear((24+2)*3+18, 256)
        else:
            ##(number of filters per object+coordinate of object)*2+question vector
            self.g_fc1 = nn.Linear((24+2)*2+18, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = paddle.empty(shape=[args.batch_size, 2])
        self.coord_oj = paddle.empty(shape=[args.batch_size, 2])

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = paddle.empty([args.batch_size, 25, 2])

        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor=paddle.to_tensor(np_coord_tensor,dtype='float32')


        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(parameters=self.parameters(), learning_rate=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        """g"""
        mb = x.shape[0]
        n_channels = x.shape[1]
        d = x.shape[2]
        # x_flat = (64 x 25 x 24)
        x_flat = paddle.reshape(x,[mb,n_channels,d*d])
        x_flat = paddle.transpose(x_flat,perm=[0,2,1])
        
        # add coordinates
        x_flat = paddle.concat([x_flat, self.coord_tensor],2)
        

        if self.relation_type == 'ternary':
            # add question everywhere
            qst = paddle.unsqueeze(qst, 1) # (64x1x18)
            qst = qst.repeat(1, 25, 1) # (64x25x18)
            qst = paddle.unsqueeze(qst, 1)  # (64x1x25x18)
            qst = paddle.unsqueeze(qst, 1)  # (64x1x1x25x18)

            # cast all triples against each other
            x_i = paddle.unsqueeze(x_flat, 1)  # (64x1x25x26)
            x_i = paddle.unsqueeze(x_i, 3)  # (64x1x25x1x26)
            x_i = x_i.repeat(1, 25, 1, 25, 1)  # (64x25x25x25x26)
            
            x_j = paddle.unsqueeze(x_flat, 2)  # (64x25x1x26)
            x_j = paddle.unsqueeze(x_j, 2)  # (64x25x1x1x26)
            x_j = x_j.repeat(1, 1, 25, 25, 1)  # (64x25x25x25x26)

            x_k = paddle.unsqueeze(x_flat, 1)  # (64x1x25x26)
            x_k = paddle.unsqueeze(x_k, 1)  # (64x1x1x25x26)
            x_k = paddle.concat([x_k, qst], 4)  # (64x1x1x25x26+18)
            x_k = x_k.repeat(1, 25, 25, 1, 1)  # (64x25x25x25x26+18)

            # concatenate all together
            x_full = paddle.concat([x_i, x_j, x_k], 4)  # (64x25x25x25x3*26+18)

            # reshape for passing through network
            x_ = paddle.reshape(x_full,shape=[mb * (d * d) * (d * d) * (d * d), 96])

        else:
            # add question everywhere
            qst = paddle.unsqueeze(qst, 1)
            qst = paddle.tile(qst,repeat_times=[1, 25, 1])
            qst = paddle.unsqueeze(qst, 2)

            # cast all pairs against each other
            x_i = paddle.unsqueeze(x_flat, 1)  # (64x1x25x26+18)
            x_i = paddle.tile(x_i,repeat_times=[1, 25, 1, 1])
            x_j = paddle.unsqueeze(x_flat, 2)  # (64x25x1x26+18)
            


            x_j = paddle.concat([x_j, qst], 3)
            x_j = paddle.tile(x_j,repeat_times=[1, 1, 25, 1])
            
            # concatenate all together
            x_full = paddle.concat([x_i,x_j],3) # (64x25x25x2*26+18)
        
            # reshape for passing through network
            x_ = paddle.reshape(x_full,[mb * (d * d) * (d * d), 70])
            
            
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        if self.relation_type == 'ternary':
            x_g = paddle.reshape(x_,[mb, (d * d) * (d * d) * (d * d), 256])
        else:
            x_g = paddle.reshape(x_,[mb, (d * d) * (d * d), 256])

        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)


class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv  = ConvInputModel()
        self.fc1   = nn.Linear(5*5*24 + 18, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(parameters=self.parameters(), learning_rate=args.lr)
        #print([ a for a in self.parameters() ] )
  
    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)
        
        x_ = paddle.concat((x, qst), 1)  # Concat question
        
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        
        return self.fcout(x_)
