local Debug, parent = torch.class('nn.Debug', 'nn.Module')

require "dump_table"

-- This module will simply always return zero.
-- Why??? Well, it's useful to null out particular
-- entries, without altering overall network architecture.

function Debug:__init(label)
	parent.__init(self)
	self.gradInput = nil
	self.output = nil
	self.label = label
end

function Debug:updateOutput(input)
	self.output = input
	fprintf(io.stderr, string.format('%s.input	', self.label))
	dump_table(io.stderr, string.format('%s.input	', self.label), input)
	fprintf(io.stderr, '\n')
	return self.output
end

function Debug:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput
	fprintf(io.stderr, string.format('%s.gradInput	', self.label))
	dump_table(io.stderr, string.format('%s.gradInput	', self.label), gradOutput)
	fprintf(io.stderr, '\n')
	return self.gradInput
end
