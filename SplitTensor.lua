local SplitTensor, parent = torch.class('nn.SplitTensor', 'nn.Module')

--[=[
So, say we have
	input = torch.randn(384,224,224) -- (Think conv3 of alexnet)
	we want to split this input so that it becomes two tensors:
	input1 = torch.randn(192, 224, 224) -- the first half
	input2 = torch.randn(192, 224, 224) -- the second half

	input1 = input:index(1, torch.range(1,192):long())
	input2 = input:index(1, torch.range(193,384):long())

	So, to accomplish this, we'd make a SplitTensor
		st = nn.SplitTensor(1, 3, {torch.range(1,192):long(), torch.range(193,384):long()})
		and we expect
		st:forward(input) to be a table of two tensors
	I think we should handle the same row being in two outputs.
]=]

function SplitTensor:__init(splitdim, ndim, indices)
	parent.__init(self)
	self.splitdim = splitdim
	self.ndim = ndim
	self.indices = indices
	self.output = nil
	self.gradInput = nil
end

function SplitTensor:updateOutput(input)
	self.output = { }
	local idx = self.splitdim + input:dim() - self.ndim
	for i, rng in ipairs(self.indices) do
		self.output[i] = input:index(idx, rng)
	end
	return self.output
end

function SplitTensor:updateGradInput(input, gradOutput)
	self.gradInput = self.gradInput or torch.Tensor():typeAs(input)
	self.gradInput:resizeAs(input):zero()
	local idx = self.splitdim + input:dim() - self.ndim

	for i, rng in ipairs(self.indices) do
		self.gradInput:indexAdd(idx, rng, gradOutput[i])
	end
	return self.gradInput
end

function SplitTensor:__tostring__()
	return string.format("%s(%d,%d,{...})", torch.type(self), self.splitdim, self.ndim)
end
