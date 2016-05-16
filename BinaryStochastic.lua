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

	if self.train
	then
--[=[
		self.output:apply(function (x) return torch.bernoulli(x) end)
]=]
		-- Pick a random value.
		self.cpubuff1 = self.cpubuff1 or torch.DoubleTensor()
		self.cpubuff1:resize(input:size())
		self.cpubuff2 = self.cpubuff2 or torch.DoubleTensor()
		self.cpubuff2:resize(input:size())

		self.cpubuff1:copy(self.rawoutput)
		self.cpubuff2:bernoulli(self.cpubuff1)

		self.output:copy(self.cpubuff2)
	else
-- Why the scale from 0..1 to 0.1..0.9? Because, if we simply multiplied by 2
-- we _could_ get 0..2 (_if_ the input was _actually_ 1 (finite numeric precision and all that)
-- So, instead, we scale 0..1 to 0.1..1.9 and just floor that.
		self.output:mul(1.8):add(0.1):floor()
	end
	return self.output
end

function BinaryStochastic:updateGradInput(input, gradOutput)
	self.output:copy(self.rawoutput)
	return nn.Sigmoid.updateGradInput(self, input, gradOutput)
end
