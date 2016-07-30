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

Not anymore. Now, you can use replacewith to replace it with whatever you like.
if you replace with -math.huge, it behaves the way you expect for logsoftmax. Or
you could replace with 0 to deal with softmax. Or you could replace with 1729 because
you like 1729.
]=]

function Censor:__init(censormask, replacewith)
	parent.__init(self)
	self.gradInput = nil
	self.output = nil

	assert(censormask:dim() == 2)
	assert(censormask:size(1) == 1)
	assert(censormask:eq(0):add(censormask:eq(1)):sum() == censormask:size(2))

	self.censormask = censormask
	self.replacewith = replacewith
end

function Censor:updateOutput(input)
	self.output = self.output or torch.Tensor(0):typeAs(input)
	self.output:resizeAs(input)
	self.output:copy(input)

	local expanded = self.censormask:typeAs(input):expandAs(input)
	local addthis = torch.zeros(expanded:size())
	addthis = addthis:typeAs(input)
	local mn = expanded:min()
	if self.replacewith == nil then
		addthis[expanded:eq(0)] = -math.huge
	else
		addthis[expanded:eq(0)] = replacewith
	end
	self.output:cmul(expanded):add(addthis)
	return self.output
end

function Censor:updateGradInput(input, gradOutput)
	self.gradInput = self.gradInput or torch.Tensor(0):typeAs(gradOutput)
	self.gradInput:resizeAs(input)
	self.gradInput:copy(gradOutput)

	local expanded = self.censormask:typeAs(input):expandAs(input)
	self.gradInput[expanded:eq(0)] = 0
	return self.gradInput
end
