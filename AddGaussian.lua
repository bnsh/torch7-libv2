local AddGaussian, parent = torch.class('nn.AddGaussian', 'nn.Module')

function AddGaussian:__init(mean, sigma)
	parent.__init(self)
	self.mean = mean
	self.sigma = sigma
	self.train = true
end

function AddGaussian:updateOutput(input)
	self.output:typeAs(input)
	self.output:resizeAs(input)
	self.output:copy(input)
	if self.train then
		self.output:add(torch.randn(input:size())*self.sigma+self.mean)
	end
	return self.output
end

function AddGaussian:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput
	return self.gradInput
end

function AddGaussian:__tostring__()
	return string.format("%s(%f,%f)", torch.type(self), self.mean, self.sigma)
end
