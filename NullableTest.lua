#! /usr/local/torch/install/bin/th

require "nn"
require "Extract"
require "Implant"
require "Nullable"

function singlerun(samples, indim, outdim)
	local fancy = nn.Linear(indim,outdim)
	local mlp = nn.Nullable(fancy)
	local criteria = nn.MSECriterion()


	local data = torch.randn(samples, indim)
	local target = torch.randn(samples, outdim)
	local indicators = torch.ByteTensor(samples):zero()
	local mode = nil
	local rnd = math.random()
	if rnd < 1/3 then
		mode = "zero"
		indicators:zero()
	elseif rnd < 2/3 then
		mode = "bernoulli"
		indicators:bernoulli()
	else
		mode = "all"
		indicators:zero()
		indicators:add(1)
	end
	local epsilon = 1e-3

	local output = mlp:forward({indicators, data})
	local err = criteria:forward(output, target)
	local gradOutput = criteria:backward(output, target)
	local gradInput = mlp:backward({indicators, data}, gradOutput)

	local empiricalGradInput = torch.zeros(samples, indim)
	for i = 1,samples do
		for j = 1,indim do
			local actual = data[{i,j}]
			local perturbedm1 = data:clone()
			perturbedm1[{i,j}] = perturbedm1[{i,j}] - epsilon
			local outputm1 = mlp:forward({indicators, perturbedm1})
			local errm1 = criteria:forward(outputm1, target)

			local perturbedp1 = data:clone()
			perturbedp1[{i,j}] = perturbedp1[{i,j}] + epsilon
			local outputp1 = mlp:forward({indicators, perturbedp1})
			local errp1 = criteria:forward(outputp1, target)

			local d = (errp1 - errm1) / (2 * epsilon)
			empiricalGradInput[{i,j}] = d
		end
	end
	local err = (gradInput[2] - empiricalGradInput)
	local rmserr = math.sqrt(err:pow(2):sum())
	return rmserr, mode
end

local epsilon = 1e-8
for j = 1,1024 do
	rmserr, mode = singlerun(128, 10, 20)
	print(string.format("Test %04d: %s: rms=%.7g %s", j, (rmserr < epsilon) and "PASS" or "FAIL!", rmserr, mode))
	assert(rmserr < epsilon)
end

