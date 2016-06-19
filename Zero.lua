local Zero, parent = torch.class('nn.Zero', 'nn.Module')

-- This module will simply always return zero.
-- Why??? Well, it's useful to null out particular
-- entries, without altering overall network architecture.

function Zero:__init(...)
	parent.__init(self)
	self.gradInput = nil
	self.output = nil
	self.args = {...}
end

function Zero:updateOutput(input)
	self.output = self.output or torch.Tensor(0):typeAs(input)
	local sz = { input:size(1) }
	for i, k in ipairs(self.args) do table.insert(sz, k) end
	self.output:resize(unpack(sz)):zero()
	return self.output
end

function Zero:updateGradInput(input, gradOutput)
	self.gradInput = self.gradInput or torch.Tensor(0):typeAs(gradOutput)
	self.gradInput:resizeAs(input):zero()
	return self.gradInput
end
