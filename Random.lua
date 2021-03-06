local Random, parent = torch.class('nn.Random', 'nn.Module')

-- This module will simply always return random values.
-- Why??? Well, it's useful to null out particular
-- entries, without altering overall network architecture.
-- In particular, images _might_ benefit from not always simply
-- outputing "zeros"

function Random:__init(sz, m, sd)
	parent.__init(self)
	self.gradInput = nil
	self.output = nil
	self.sz = sz
	self.train = true
	self.m = m or 0
	self.sd = sd or 1
end

function Random:updateOutput(input)
	self.output = self.output or torch.Tensor(0):typeAs(input)
	local sz = { input:size(1) }
	for i, k in ipairs(self.sz) do table.insert(sz, k) end
	self.output:resize(unpack(sz))
	if self.train then
		self.output:normal():mul(self.sd):add(self.m) -- This is badly named. But, this generates random normally distributed values, it is _not_ doing the "norm" of the vector.
	else
		self.output:zero()
	end
	return self.output
end

function Random:updateGradInput(input, gradOutput)
	self.gradInput = self.gradInput or torch.Tensor(0):typeAs(gradOutput)
	self.gradInput:resizeAs(input):zero()
	return self.gradInput
end
