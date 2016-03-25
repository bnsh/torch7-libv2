local BinaryStochastic = torch.class('nn.BinaryStochastic', 'nn.Sigmoid')

function BinaryStochastic:__init()
	nn.Sigmoid.__init(self)
	self.rawoutput = torch.Tensor()
end

function BinaryStochastic:updateOutput(input)
	nn.Sigmoid.updateOutput(self, input)
	self.rawoutput = self.rawoutput or input.new()
	self.rawoutput:resizeAs(input)

-- No matter what, copy the rawoutput.
	self.rawoutput:copy(self.output)

	if not self.train
	then
		self.output:mul(2):floor()
		self.output[self.output:lt(0)] = 0
		self.output[self.output:gt(1)] = 1
	else
		-- Pick a random value.
		self.output:apply(function (x) return torch.bernoulli(x) end)
	end
	return self.output
end

function BinaryStochastic:updateGradInput(input, gradOutput)
	self.output:copy(self.rawoutput)
	return nn.Sigmoid.updateGradInput(self, input, gradOutput)
end
