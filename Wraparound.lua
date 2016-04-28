local Wraparound, parent = torch.class('nn.Wraparound', 'nn.Module')

function Wraparound:__init(mini, maxi)
	parent.__init(self)
	self.mini = mini
	self.maxi = maxi
end

function Wraparound:updateOutput(input)
	self.output:typeAs(input)
	self.output:resizeAs(input)
	self.output:copy(input)
	self.output:add(-self.mini)
	self.output:mul(1.0/(self.maxi-self.mini))
	self.output:floor()
	self.output:mul(self.mini-self.maxi)
	self.output:add(input)
	return self.output
end

function Wraparound:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput
	return self.gradInput
end

function Wraparound:__tostring__()
	return string.format("%s(%f,%f)", torch.type(self), self.mini, self.maxi)
end
