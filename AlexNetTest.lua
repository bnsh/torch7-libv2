#!  /usr/local/torch/install/bin/th

require "cutorch"
require "nn"
require "loadcaffe"
require "nnio"
require "lfs"
local nnio_util = require "nnio_util"
require "image"
require "AlexNet"
require "fprintf"

--[=[
Run this before.
rsync -av --progress /usr/local/caffe/models/bvlc_alexnet/ /tmp/an/
and, dump the bvlc_alexnet model with ../caffe2lua/dump_alexnet
]=]

local function read_labels(fn)
	local rv = { }
	local fh = io.open(fn, "r")
	if fh ~= nil then
		for line in fh:lines() do
			table.insert(rv, line)
		end
		fh:close()
	end
	return rv
end

local function untangle(mlp, lo,hi)
	local rv = nn.Sequential()
	for i = lo,hi do
		rv:add(mlp.modules[i])
	end
	return rv
end

cutorch.setDevice(cutorch.getDeviceCount())
local mine = nn.AlexNet(nil, false):add(nn.Squeeze()):add(nn.LogSoftMax())
local sergey = loadcaffe.load("/tmp/an/deploy.prototxt", "/tmp/an/bvlc_alexnet.caffemodel", "cudnn")

mine:forward(torch.randn(16,3,227,227))

local batchsz = 10
local mappings = {
	{ "conv1", mine.modules[1].modules[1],sergey.modules[1], torch.LongStorage({batchsz,3,227,227}) },
	{ "conv2", untangle(mine.modules[2],1,3),sergey.modules[5], torch.LongStorage({batchsz,96,27,27}) },
	{ "conv3", mine.modules[3].modules[1],sergey.modules[9], torch.LongStorage({batchsz,256,13,13}) },
	{ "conv4", untangle(mine.modules[4],1,3),sergey.modules[11], torch.LongStorage({batchsz,384,13,13}) },
	{ "conv5", untangle(mine.modules[5],1,3),sergey.modules[13], torch.LongStorage({batchsz,384,13,13}) },
	{ "fc6", untangle(mine.modules[6],1,2),untangle(sergey,16,17), torch.LongStorage({batchsz,256,6,6}) },
	{ "fc7", mine.modules[7].modules[1],sergey.modules[20], torch.LongStorage({batchsz,4096}) },
	{ "fc8", mine.modules[8].modules[1],sergey.modules[23], torch.LongStorage({batchsz,4096}) }
}

nnio_util.restore("/tmp/bnet", mine)
mine:evaluate()
sergey:evaluate()

local function probe(label, mlp1, mlp2, sz)
	local success = true
	for i = 1, 16 do
		local rnd = torch.randn(sz)
		local output1 = mlp1:forward(rnd)
		local output2 = mlp2:forward(rnd:cuda()):double()
		local rms = math.sqrt((output1-output2):pow(2):sum())
		local status = (((rms < 1e-3) and "PASS") or "FAIL")
		if (rms >= 1e-3) then
			print(string.format("%s: %2d: %s: rms=%.7g", label, i, status, rms))
			success = false
		end
	end
	return success
end

--[=[
for i, mapping in ipairs(mappings) do
	if probe(unpack(mapping)) then print(string.format("%s: PASS", mapping[1])) end
end
--]=]

local labels = read_labels("/usr/local/caffe/data/ilsvrc12/synset_words.txt")
local qq = image.load("cat.jpg", 3, 'double')
local channels = qq:size(1)
local height = qq:size(2)
local width = qq:size(3)
local smaller = ((height < width) and height) or width
qq = image.crop(qq, "c", smaller, smaller)
qq = image.scale(qq, 227, 227):add(-0.5):mul(255)

local preds = mine:forward(qq)
local vals, idxs = torch.sort(preds, 1, true)

for i = 1,5 do
	fprintf(io.stdout, "%.7f%%	%d	%s\n", 100*math.exp(vals[{i}]), idxs[{i}], labels[idxs[{i}]])
end


