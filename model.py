import torch,torchvision, torch.nn as nn
import torchvision.transforms.functional as fn



class Encoder(nn.Module):
            def __init__(self,in_channel, out_channel, stride, pool_kernel, *args, **kwargs) :
                super().__init__(*args, **kwargs)
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=7, padding=3, stride=stride),
                    nn.Dropout2d(0.4),
                    nn.BatchNorm2d(out_channel),
                    nn.LeakyReLU(0.2, inplace=True)
                )
                self.pool = nn.MaxPool2d(kernel_size=pool_kernel)
            def forward(self, input):
                x = self.conv(input)
                p = self.pool(x)
                # both x and p has the same size with pooling_kernel of 1
                return x, p

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, stride, *args, **kwargs) :
        super().__init__(*args, **kwargs)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=7, padding=3, stride=stride ),
            nn.Dropout2d(0.4),
            nn.ConvTranspose2d(out_channel, out_channel, kernel_size=7, padding=3 )
        )
        # self.conv = Encoder(out_channel+out_channel, out_channel, stride=1, pool_kernel=1)

    def forward(self, input, skip=None):
        if skip != None:
            if input.shape != skip.shape:
                skip = fn.resize(skip, input.shape[2:])
            input = input + skip
            
        x = self.deconv(input)
        # x = self.conv(x)
        return x

class UnBlur_Down(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.e1 = Encoder(3, 16, stride=2,  pool_kernel=1)

        self.e2 = Encoder(16, 32, stride=2,  pool_kernel=1)
        
        self.e3 = Encoder(32, 64, stride=2,  pool_kernel=1)
        # 256 -> 128

        self.e4 = Encoder(64, 128, stride=2,  pool_kernel=1)
        # 128 -> 64

        self.e5 = Encoder(128, 256, stride=2, pool_kernel=1)
        # 64 -> 32

        ####### BottleNeck ######
        self.d1 = Decoder(256, 128, stride=2)
        self.d2 = Decoder(128, 64, stride=2)
        self.d3 = Decoder(64, 32, stride=2)
        self.d4 = Decoder(32, 16, stride=2)
        self.d5 = Decoder(16, 3, stride=2)

        self.relu = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(8)
    
    def forward(self, input):
        s1, p1 = self.e1(input)
        s2, p2 = self.e2(s1)
        s3, p3 = self.e3(s2)
        s4, p4 = self.e4(s3)
        s5, p5 = self.e5(s4)

        d1 = self.d1(s5)
        # d1 = self.bn1(d1[0])
        d1 = self.relu(d1)

        d2 = self.d2(d1, p4)
        # d2 = self.bn2(d2)
        d2 = self.relu(d2)

        d3 = self.d3(d2, p3)
        # d3 = self.bn3(d3)
        d3 = self.relu(d3)

        d4 = self.d4(d3, p2)
        # d4 = self.bn4(d4)
        d4 = self.relu(d4)


        d5 = self.d5(d4, p1)
        # d5 = self.bn5(d5[0])
        d5 = self.relu(d5)



        return d5    


class UnBlur_Up(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e1 = Encoder(3, 32, stride=2, pool_kernel=1)
        ## image_size -> image_size / 2
        
        self.e2 = Encoder(32, 64, stride=2, pool_kernel=1)
        # image_size / 2 -> image_size / 4

        ####### BottleNeck ######
        self.d1 = Decoder(64,  32, stride=2)
        self.d2 = Decoder(32,   3,  stride=2)
        self.relu = nn.LeakyReLU(0.2)

    
    def forward(self, input):
        
        s1, p1 = self.e1(input)
        s2, p2 = self.e2(p1)

        d1 = self.d1(s2)

        d1 = self.relu(d1)

        d2 = self.d2(d1, p1)
        d2 = self.relu(d2)

        return d2
    
def create_model(device):
    up_model    = UnBlur_Up().to(device)
    down_model  = UnBlur_Down().to(device)

    loss_function = nn.MSELoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    optimizer_params = [
        {'params':up_model.parameters()},
        {'params':down_model.parameters()},
    ]
    optimizer = torch.optim.Adam(optimizer_params,lr=0.0001,weight_decay=0.4)

    return up_model, down_model, loss_function, optimizer

