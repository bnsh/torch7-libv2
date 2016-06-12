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
	local indicators = torch.ByteTensor(samples):bernoulli()
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
	return rmserr
end

local epsilon = 1e-8
for j = 1,1024 do
	rmserr = singlerun(128, 10, 20)
	print(string.format("Test %04d: rms=%.7g %s", j, rmserr, (rmserr < epsilon) and "PASS" or "FAIL!"))
	assert(rmserr < epsilon)
end

