local Cos, parent = torch.class('nn.Cos', 'nn.Module')

function Cos:__init()
	parent.__init(self)
end

function Cos:updateOutput(input)
	return self.output:cos(input)
end

function Cos:updateGradInput(input, gradOutput)
	return self.gradInput:cmul(-torch.sin(input), gradOutput)
end
