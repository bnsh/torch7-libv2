local Sin, parent = torch.class('nn.Sin', 'nn.Module')

function Sin:__init()
	parent.__init(self)
end

function Sin:updateOutput(input)
	return self.output:sin(input)
end

function Sin:updateGradInput(input, gradOutput)
	return self.gradInput:cmul(torch.cos(input), gradOutput)
end
