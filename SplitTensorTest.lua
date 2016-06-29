#! /usr/local/torch/install/bin/th

require "cutorch"
require "nn"
require "SplitTensor"

function single_round()
	-- So, construct a random dimension
	local dim = 1+math.random(3) -- This should be in the range 2..4
	local sz = { }
	for q = 1, dim do
		sz[q] = 1+math.random(5) -- This should be in the range 2..6
	end

-- So, our maximum data size should be 6^4 = 1296 not so bad.
	data = torch.randn(torch.LongStorage(sz))

	local splitdim = math.random(dim)
-- Now, create a bunch of splits.
	local nsplits = 1+math.random(3) -- This should be in the range 2..4
	local split = { }
	for i = 1, nsplits do
		local cc = math.random(1,sz[splitdim])
		split[i] = torch.randperm(sz[splitdim])[{{1,cc}}]:long()
	end

	local st = nn.SplitTensor(splitdim, dim, split)
	local output = st:forward(data)
	local criteria = nn.ParallelCriterion()
	local target = { }
	for i, v in ipairs(output) do
		target[i] = torch.randn(v:size())
		criteria:add(nn.MSECriterion())
	end

	local output = st:forward(data)
	local err = criteria:forward(output, target)
	local gradOutput = criteria:backward(output, target)
	local purportedGradInput = st:backward(data, gradOutput)

-- Sure. So you say. We shall see.
	local epsilon = 1e-3

	local resizeddata = data:view(-1)
	local actualGradInput = torch.zeros(resizeddata:size(1))
	for i = 1, resizeddata:size(1) do
		local pristine = resizeddata[i]
		resizeddata[i] = pristine - epsilon
		local output = st:forward(resizeddata:viewAs(data))
		local errm1 = criteria:forward(output, target)

		resizeddata[i] = pristine + epsilon
		local output = st:forward(resizeddata:viewAs(data))
		local errp1 = criteria:forward(output, target)

		resizeddata[i] = pristine
		actualGradInput[i] = (errp1 - errm1) / (2 * epsilon)
	end

	local rms = math.sqrt(((purportedGradInput:view(-1) - actualGradInput)):pow(2):sum())
	return rms
end
	
function test_splittensor(iterations)
	for i = 1, iterations do
		local rms = single_round()
		print(string.format("i=%d: %s: RMS=%.7g", i, (((rms < 1e-8) and "PASS") or "FAIL"), rms))
	end
end

test_splittensor(1024)
