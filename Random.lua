local Random, parent = torch.class('nn.Random', 'nn.Module')

-- This module will simply always return random values.
-- Why??? Well, it's useful to null out particular
-- entries, without altering overall network architecture.
-- In particular, images _might_ benefit from not always simply
-- outputing "zeros"

function Random:__init(...)
	parent.__init(self)
	self.gradInput = nil
	self.output = nil
	self.args = {...}
end

function Random:updateOutput(input)
	self.output = self.output or torch.Tensor(0):typeAs(input)
	local sz = { input:size(1) }
	for i, k in ipairs(self.args) do table.insert(sz, k) end
	self.output:resize(unpack(sz))
	if self.train then
		self.output:copy(torch.randn(unpack(sz)))
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
