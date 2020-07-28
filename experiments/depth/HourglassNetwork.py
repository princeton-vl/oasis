import torch
from torch import nn
from torch.autograd import Variable
from inception import inception

class Channels1(nn.Module):
	def __init__(self):
		super(Channels1, self).__init__()
		self.list = nn.ModuleList()
		self.skip = nn.Sequential(
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]]),
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]])
				)
			
		self.ap = nn.AvgPool2d(2)
		self.inc1 = inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]])
		self.inc2 = inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]])
		self.inc3 = inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]])
		self.up = nn.UpsamplingNearest2d(scale_factor=2)

	def forward(self,x):
		out = self.ap(x)
		out = self.inc1(out)
		out = self.inc2(out)
		feat = self.inc3(out)
		depth = self.up(feat)

		return self.skip(x)+depth, feat

class Channels2(nn.Module):
	def __init__(self):
		super(Channels2, self).__init__()
		self.list = nn.ModuleList()
		self.bridge = nn.Sequential(
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]), 
				inception(256, [[64], [3,64,64], [7,64,64], [11,64,64]])
				)
			
		self.ap = nn.AvgPool2d(2)
		self.inc1 = inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]])
		self.inc2 = inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]])
		self.channel = Channels1()
		self.inc3 = inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]])
		self.inc4 = inception(256, [[64], [3,64,64], [7,64,64], [11,64,64]])
		self.up = nn.UpsamplingNearest2d(scale_factor=2)

	def forward(self,x):
		out = self.ap(x)
		out = self.inc1(out)
		out = self.inc2(out)
		depth, feat = self.channel(out)
		depth = self.inc3(depth)
		depth = self.inc4(depth)
		depth = self.up(depth)

		return self.bridge(x)+depth, feat

class Channels3(nn.Module):
	def __init__(self):
		super(Channels3, self).__init__()
		self.list = nn.ModuleList()
		
		self.ap = nn.AvgPool2d(2)
		self.inc1 = inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]])
		self.inc2 = inception(128, [[64], [3,32,64], [5,32,64], [7,32,64]])
		self.channel = Channels2()
		self.inc3 = inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]])
		self.inc4 = inception(256, [[32], [3,32,32], [5,32,32], [7,32,32]])
		self.up = nn.UpsamplingNearest2d(scale_factor=2)
			
		self.bridge = nn.Sequential(
				inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]]), 
				inception(128, [[32], [3,64,32], [7,64,32], [11,64,32]])
				)
			

	def forward(self,x):
		out = self.ap(x)
		out = self.inc1(out)
		out = self.inc2(out)
		depth, feat = self.channel(out)
		depth = self.inc3(depth)
		depth = self.inc4(depth)
		depth = self.up(depth)
		return self.bridge(x) + depth, feat

class Channels4(nn.Module):
	def __init__(self):
		super(Channels4, self).__init__()

		self.ap = nn.AvgPool2d(2)
		self.inc1 = inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]])
		self.inc2 = inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]])
		self.channel = Channels3()
		self.inc3 = inception(128, [[32], [3,64,32], [5,64,32], [7,64,32]])
		self.inc4 = inception(128, [[16], [3,32,16], [7,32,16], [11,32,16]])
		self.up = nn.UpsamplingNearest2d(scale_factor=2)

		self.bridge = inception(128, [[16], [3,64,16], [7,64,16], [11,64,16]])

	def forward(self,x):
		out = self.ap(x)
		out = self.inc1(out)
		out = self.inc2(out)
		depth, feat = self.channel(out)
		depth = self.inc3(depth)
		depth = self.inc4(depth)
		depth = self.up(depth)

		return self.bridge(x)+depth, feat


class HourglassNetwork(nn.Module):
	def __init__(self):
		super(HourglassNetwork, self).__init__()


		self.min_focal = 250.0
		print( "===================================================")
		print( "Using HourglassNetwork")
		print( "Min focal = %g" % self.min_focal)
		print( "===================================================")

		
		self.conv1 = nn.Conv2d(3,128,7,padding=3)
		self.conv2 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU(True)
		self.channel = Channels4()
		self.conv3 = nn.Conv2d(64,1,3,padding=1)
		self.relu2 = nn.ReLU(True)

		self.avgpool = nn.AdaptiveAvgPool2d((3, 4))
		self.fc_disp = nn.Linear(3072, 1)
		self.fc_focal = nn.Linear(3072,1)

		self.relu = nn.ReLU(inplace=True)

	def forward(self,x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.relu1(out)
		depth, feat = self.channel(out)
		depth = self.conv3(depth)
		depth = self.relu2(depth)


		#branch 2: 
		feat = self.avgpool(feat)               
		feat = feat.view(-1, 3072)


		log_disp = self.fc_disp(feat)        # [batch_size, 1]   
		disp = torch.exp(log_disp)      # take the exponential
		disp = disp.unsqueeze(-1).unsqueeze(-1)
		disp = disp.expand(-1, -1, depth.shape[2], depth.shape[3])
		

		log_focal = self.fc_focal(feat) # [batch_size, 1]
		focal = torch.exp(log_focal) + self.min_focal    # take the exponential       

		depth = depth + disp            # [batch_size, 1, H, W]
		depth = self.relu(depth) + 1e-8

		return depth, focal

