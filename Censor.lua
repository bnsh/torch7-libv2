local Censor, parent = torch.class('nn.Censor', 'nn.Module')

--[=[
This will set particular columns in the input to zero. Why would anyone ever want to do this??
Say you have a LogSoftMax, but you want to censor out particular values. So, you have
a classifier that can classify things as aardvarks, baseballs, cats, dogs and elephants.
You _know_ we're looking at animals. (Or perhaps for whatever reason, (as is my actual case)
the powers that be, just don't want to see baseballs.) You can put a censor like so before
the LogSoftMax: 
	local mlp = nn.Sequential()
	mlp:add(nn.Censor(torch.ByteTensor({{1,0,1,1,1}}))
	mlp:add(nn.LogSoftMax())

and now, you'll have sensible results for p(x=aardvark|x is _NOT_ baseball)

So, we're *NOT* zeroing values out. We're making them very very negative. Because
censor is designed to work _specifically_ with LogSoftMax!
]=]

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
	self.output:cmul(expanded):add(addthis)
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
