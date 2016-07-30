local Censor, parent = torch.class('nn.Censor', 'nn.Module')

function Censor:__init(censortable)
	parent.__init(self)
	self.gradInput = nil
	self.output = nil

	assert(censortable:dim() == 2)
	assert(censortable:size(1) == 1)
	assert(censortable:eq(0):add(censortable:eq(1)):sum() == censortable:size(2))

	self.censortable = censortable
end

function Censor:updateOutput(input)
	self.output = self.output or torch.Tensor(0):typeAs(input)
	self.output:resizeAs(input)
	self.output:copy(input)

	local expanded = self.censortable:typeAs(input):expandAs(input)
	self.output:cmul(expanded)
	return self.output
end

function Censor:updateGradInput(input, gradOutput)
	self.gradInput = self.gradInput or torch.Tensor(0):typeAs(gradOutput)
	self.gradInput:resizeAs(input)
	self.gradInput:copy(gradOutput)

	local expanded = self.censortable:typeAs(input):expandAs(input)
	self.gradInput:cmul(expanded)
	return self.gradInput
end
