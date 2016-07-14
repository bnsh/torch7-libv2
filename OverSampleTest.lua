#! /usr/local/torch/install/bin/th

require "cutorch"
require "nn"
require "OverSample"

function single_round()
	local inputs = math.random(3,5)
	local channels = math.random(1,3)
	local szx = math.random(8,16)
	local szy = math.random(8,16)
	local patchx = math.random(6,10)
	local patchy = math.random(6,10)

-- So, what will our output size be? channelsxpatchxxpatchy _unless_ patchx > szx or patchy > szy
	local tx = (patchx < szx) and patchx or szx
	local ty = (patchy < szy) and patchy or szy
	local outputsz = channels * ty * tx
	local lsmoutsz = 16

	local mlp = nn.Sequential()
	mlp:add(nn.OverSample(patchx, patchy))
	mlp:add(nn.CAddTable())
	mlp:add(nn.Reshape(outputsz))
	mlp:add(nn.Linear(outputsz, lsmoutsz))
	mlp:add(nn.LogSoftMax())

-- Now make a random "image"
	local data = torch.randn(inputs,channels, szy, szx)
	local target = torch.ones(inputs)
	local output = mlp:forward(data)
	assert(not output:ne(output):any())

	local criteria = nn.ClassNLLCriterion()
	local err = criteria:forward(output, target)
	local gradOutput = criteria:backward(output, target)
	local purportedGradInput = mlp:backward(data, gradOutput)

	local epsilon = 1e-3
	local resizeddata = data:view(-1)
	local actualGradInput = torch.zeros(resizeddata:size(1))
	for i = 1, resizeddata:size(1) do
		local pristine = resizeddata[i]
		resizeddata[i] = pristine - epsilon
		local output = mlp:forward(resizeddata:viewAs(data))
		local errm1 = criteria:forward(output, target)

		resizeddata[i] = pristine + epsilon
		local output = mlp:forward(resizeddata:viewAs(data))
		local errp1 = criteria:forward(output, target)

		resizeddata[i] = pristine
		actualGradInput[i] = (errp1 - errm1) / (2 * epsilon)
	end

	local rms = math.sqrt(((purportedGradInput:view(-1) - actualGradInput)):pow(2):sum())
	return rms
end
	
function test_oversample(iterations)
	for i = 1, iterations do
		local rms = single_round()
		print(string.format("i=%d: %s: RMS=%.7g", i, (((rms < 1e-4) and "PASS") or "FAIL"), rms))
	end
end

test_oversample(1024)

--[=[
]=]
