require "nn"
require "cunn"
require "cudnn"
require "Censor"
require "fprintf"

local function verify(batchsz, sz)
	local mask = torch.ByteTensor(1, sz):bernoulli()
	local censor = nn.Censor(mask)
	local input = torch.randn(batchsz, sz)
	local target = torch.randn(batchsz, sz)
	local criteria = nn.MSECriterion()
	
	local output = censor:forward(input)
	local err = criteria:forward(output, target)
	local gradOutput = criteria:backward(output, target)
	local purportedGradInput = censor:backward(input, gradOutput)
	local actualGradInput = torch.zeros(batchsz, sz)
	local epsilon = 1e-3

	-- Great. Now, let's compute it with finite differences.
	for i = 1, batchsz do
		for j = 1, sz do
			local pristine = input[{i,j}]

			input[{i,j}] = pristine + epsilon
			local errp1 = criteria:forward(censor:forward(input), target)

			input[{i,j}] = pristine - epsilon
			local errm1 = criteria:forward(censor:forward(input), target)

			actualGradInput[{i,j}] = (errp1 - errm1) / (2 * epsilon)

			input[{i,j}] = pristine
		end
	end

	local errsquared = math.sqrt((purportedGradInput - actualGradInput):pow(2):sum())
	return errsquared
end

local function main()
	local iters = 1024
	local batchsz = 32
	local sz = 32
	for i = 1, iters do
		local rms = verify(batchsz, sz)
		local status = ((rms < 1e-3) and "PASS") or "FAIL"
		fprintf(io.stderr, "%d: %s: %.7g\n", i, status, rms)
		assert(rms < 1e-3)
	end
end

main()
