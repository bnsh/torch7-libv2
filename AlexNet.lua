require "SplitTensor"
require "SpatialDropoutX21"
local AlexNet, parent = torch.class('nn.AlexNet', 'nn.Sequential')

--[=[
Actually, this is basically verbatim from
	https://github.com/soumith/imagenet-multiGPU.torch/blob/master/models/alexnet.lua
I've just removed bits and pieces and parametrized the size.
Also, I'm not sure what the purpose of nn.Threshold(0, 1e-6) is. It seems to be
just doing a ReLU... But, with an odd kink in the middle from {0,1e-6}


Actually, I modified it a bit to make it fit more closely deploy.prototxt from
	https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet

Ideally, loading from this: http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
and nn.Sequential():add(nn.AlexNet(1000)):add(nn.LogSoftMax()) will produce the
same outputs. We shall see.

Actually, now at this point, this is just my own re-implementation of AlexNet. Also,
I changed the "fc"'s to "convs"... It produces the same output.
]=]

local function conv1(dropout, transfer)
	-- conv1 takes the image as input and outputs a feature map of size 96
	local rv = nn.Sequential()
	rv:add(nn.SpatialConvolution(3,96,11,11,4,4))
	rv:add(transfer())
	rv:add(nn.SpatialCrossMapLRN(5,0.0001,0.75))
	rv:add(nn.SpatialMaxPooling(3,3,2,2))
	rv:add(nn.SpatialDropoutX21(dropout))

	return rv
end


local function conv2(dropout, transfer)
	-- conv2 takes the 96 features, splits it into 2 48 features and runs over
	-- each of those to produce 2 128 features and then joins them both again
	-- as 256 features.
	local rv = nn.Sequential()
	rv:add(nn.SplitTensor(1,3,{torch.range(1,48):long(), torch.range(49,96):long()}))
	rv:add(nn.ParallelTable():
		add(nn.SpatialConvolution(48,128,5,5,1,1,2,2)):
		add(nn.SpatialConvolution(48,128,5,5,1,1,2,2))
	)
	rv:add(nn.JoinTable(1,3))
	rv:add(transfer())
	rv:add(nn.SpatialCrossMapLRN(5,0.0001,0.75))
	rv:add(nn.SpatialMaxPooling(3,3,2,2))
	rv:add(nn.SpatialDropoutX21(dropout))

	return rv
end

local function conv3(dropout, transfer)
	local rv = nn.Sequential()
	-- conv3 is _not_ doing the splitting business.
	rv:add(nn.SpatialConvolution(256,384,3,3,1,1,1,1))
	rv:add(transfer())
	rv:add(nn.SpatialDropoutX21(dropout))
	return rv
end

local function conv4(dropout, transfer)
	-- conv4 takes the 384 features, splits it into 2 192 features and runs over
	-- each of those to produce 2 192 features and then joins them both again
	-- as 384 features.
	local rv = nn.Sequential()
	rv:add(nn.SplitTensor(1,3,{torch.range(1,192):long(), torch.range(193,384):long()}))
	rv:add(nn.ParallelTable():
		add(nn.SpatialConvolution(192,192,3,3,1,1,1,1)):
		add(nn.SpatialConvolution(192,192,3,3,1,1,1,1))
	)
	rv:add(nn.JoinTable(1,3))
	rv:add(transfer())
	rv:add(nn.SpatialDropoutX21(dropout))

	return rv
end

local function conv5(dropout, transfer)
	-- conv5 takes the 384 features, splits it into 2 192 features and runs over
	-- each of those to produce 2 128 features and then joins them both again
	-- as 256 features.
	local rv = nn.Sequential()
	rv:add(nn.SplitTensor(1,3,{torch.range(1,192):long(), torch.range(193,384):long()}))
	rv:add(nn.ParallelTable():
		add(nn.SpatialConvolution(192,128,3,3,1,1,1,1)):
		add(nn.SpatialConvolution(192,128,3,3,1,1,1,1))
	)
	rv:add(nn.JoinTable(1,3))
	rv:add(transfer())
	rv:add(nn.SpatialMaxPooling(3,3,2,2))
	rv:add(nn.SpatialDropoutX21(dropout))

	return rv
end

local function fc6(dropout, transfer, bottleneck_sz)
	-- fc6 takes as input a 256x6x6 tensor and outputs a bottleneck_sz tensor
	local rv = nn.Sequential()
	rv:add(nn.Reshape(256*6*6))
	rv:add(nn.Linear(256*6*6, bottleneck_sz))
	rv:add(transfer())
	rv:add(nn.Dropout(dropout))
	return rv
end

local function fc7(bottleneck_sz, dropout, transfer)
	-- fc7 takes as input a bottleneck_sz tensor and outputs a 4096 tensor
	local rv = nn.Sequential()
	rv:add(nn.Linear(bottleneck_sz, 4096))
	rv:add(transfer())
	rv:add(nn.Dropout(dropout))
	return rv
end

local function fc8(sz, dropout, transfer)
	-- fc8 takes as input a 4096 tensor and outputs a sz tensor
	local rv = nn.Sequential()
	rv:add(nn.Linear(4096, sz))
	rv:add(nn.Reshape(sz,1,1)) -- Just so that everything seems the same whether we
				   -- use conv8 or fc8
	return rv
end

local function conv6(dropout, transfer, bottleneck_sz)
	-- conv6 takes as input a 256x6x6 tensor and outputs a bottleneck_szx1x1 tensor
	local rv = nn.Sequential()
	rv:add(nn.Identity()) -- Just so that weights are stored in the same place as fc6
	rv:add(nn.SpatialConvolution(256,bottleneck_sz,6,6,1,1))
	rv:add(transfer())
	rv:add(nn.SpatialDropoutX21(dropout))
	return rv
end

local function conv7(bottleneck_sz,dropout, transfer)
	-- conv7 takes as input a bottleneck_szx1x1 tensor and outputs a 4096x1x1 tensor
	local rv = nn.Sequential()
	rv:add(nn.SpatialConvolution(bottleneck_sz, 4096, 1,1, 1,1))
	rv:add(transfer())
	rv:add(nn.SpatialDropoutX21(dropout))
	return rv
end

local function conv8(sz, dropout, transfer)
	-- conv8 takes as input a 4096x1x1 tensor and outputs a szx1x1 tensor
	local rv = nn.Sequential()
	rv:add(nn.SpatialConvolution(4096, sz, 1,1, 1,1))
	return rv
end

function defaults(params, def)
	params = params or { }
	for k, v in pairs(def) do
		params[k] = params[k] or v
	end
	return params
end

function AlexNet:__init(params, fullyconvolutional)
	parent.__init(self)

	fullyconvolutional = fullyconvolutional or false
	params = params or { }
	params.general = defaults(params.general, {
			dropout=0.5,
			transfer=nn.ReLU
		})
	params.bottleneck = defaults(params.bottleneck, {
			sz=4096,
			dropout=0.5,
			transfer=nn.ReLU
		})
	params.output = defaults(params.output, {
			sz=1000,
			dropout=0.5,
			transfer=nn.ReLU
		})
	                                                                                                     -- 3x227x227
	self:add(conv1(params.general.dropout, params.general.transfer))                                     -- 96x27x27
	self:add(conv2(params.general.dropout, params.general.transfer))                                     -- 256x13x13
	self:add(conv3(params.general.dropout, params.general.transfer))                                     -- 384x13x13
	self:add(conv4(params.general.dropout, params.general.transfer))                                     -- 384x13x13
	self:add(conv5(params.general.dropout, params.general.transfer))                                     -- 256x6x6
	if fullyconvolutional then
		self:add(conv6(params.bottleneck.dropout, params.bottleneck.transfer, params.bottleneck.sz)) -- bottleneck_sz
		self:add(conv7(params.bottleneck.sz, params.general.dropout, params.general.transfer))       -- 4096
		self:add(conv8(params.output.sz, params.output.dropout, params.output.transfer))             -- outputsz
	else
		self:add(fc6(params.bottleneck.dropout, params.bottleneck.transfer, params.bottleneck.sz))   -- bottleneck_sz
		self:add(fc7(params.bottleneck.sz, params.general.dropout, params.general.transfer))         -- 4096
		self:add(fc8(params.output.sz, params.output.dropout, params.output.transfer))               -- outputsz
	end
end
