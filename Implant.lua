local Implant, parent = torch.class('nn.Implant', 'nn.Module')

--[=[
So.
	Our input is a table {indicators, data}
	This module is in _principle_ the opposite of nn.Extract. In reality, of course,
	Implant(Extract(x)) will not be x, because, the nature of extracting pieces
		of x, will mean that the implanter can't reconstruct those missing pieces.
	So, for instance
		Implant({1,0,1},Extract({1,0,1},{{1,2,3},{4,5,6},{7,8,9}})) would return
			{{1,2,3},{0,0,0},{7,8,9}}

	(Of course, this is precisely what we want, and is the point of this module to begin
	with.)
]=]

function Implant:__init()
	parent.__init(self)
	self.output = nil
end

function Implant:updateOutput(input)
	local indicators = input[1]
	local data = input[2]
	local indicatorsz = torch.ones(data:dim())
	indicatorsz[1] = indicators:size(1)
	local outputsz = data:size()
	outputsz[1] = indicators:size(1)
-- What should the output sz be? It should be indicators:size(1) but data:size(2-)
	local expandedindicators = indicators:gt(0.5):resize(torch.LongStorage(indicatorsz:totable())):expand(outputsz)

-- Now, we have to put back the rows that we took out.
	self.output = self.output or torch.Tensor(0):typeAs(data)

	self.output:resize(torch.LongStorage(outputsz:totable())):zero()
	self.output:maskedCopy(expandedindicators, data)
	return self.output
end

function Implant:updateGradInput(input, gradOutput)
	local indicators = input[1]

	self.gradInput = {
		torch.zeros(indicators:size()):typeAs(indicators),
		gradOutput:index(1,torch.range(1,indicators:numel())[indicators:gt(0.5)]:long())
	}
	return self.gradInput
end
