#! /usr/local/torch/install/bin/th

require "fprintf"
require "nn"
require "kldivergence"

--[=[
What are some good test cases?

target=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
output=[  a,   b,   c,   d,   e,   f,   g,   h,   i,   j]
expect=Sum[0.1 (output_i - log(0.1))] =  0.1 Sum[output_i] - Log[0.1] ?

target=[  a,   b,   c,   d,   e,   f,   g,   h,   i,   j]
output=[  k,   k,   k,   k,   k,   k,   k,   k,   k,   k]
expect=Sum[t_i (k - Log[t_i])] = k Sum[t_i] - Sum[t_i Log[t_i]] = k - Sum[t_i Log[t_i]]
	(Because Sum[t_i] == 1 since these are probabilities.)

Then, of course, just pass a bunch directly to KLDivCriterion and check the sums.
Maybe I'll just do that instead.
]=]

local function random_distribution(samples, sz, datap)
	local raw = torch.zeros(samples, sz):uniform(0,1)
-- Let's set a bunch of these to zero for good measure.
	local bernoulli = torch.zeros(samples, sz):bernoulli(datap)
	raw:cmul(bernoulli)
	raw:cdiv(raw:sum(2):repeatTensor(1, raw:size(2)))
	return raw
end

local function single_iteration()
	local criterion = nn.DistKLDivCriterion()
	criterion.sizeAverage = false
	local samples = 1024
	local sz = 16
	local output = random_distribution(samples,sz, 1):log()
	local target = random_distribution(samples,sz, 0.9)
	local expect = criterion:forward(output, target)
	local actual = kldivergence(output, target):sum()

	local err = math.sqrt((expect - actual) * (expect - actual))
	return err
	
end

function testkldivergence()
	for i = 1, 100 do
		local err = single_iteration()

		local status = "fail"
		if err < 0.0001 then
			status = "pass"
		end
		fprintf(io.stderr, "%5d: %s: %.7f\n", i, status, err)
	end
end

testkldivergence()
