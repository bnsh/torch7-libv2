local SpatialDropoutX21, Parent = torch.class('nn.SpatialDropoutX21', 'nn.Module')

--[=[
	The real difference between this and SpatialDropout, is that SpatialDropout
	multiplies by (1-self.p) at evaluation time, and I divide by (1-self.p) at
	training time. (Closer to v2 on regular nn.Dropout
]=]

function SpatialDropoutX21:__init(p)
   Parent.__init(self)
   self.p = p or 0.5
   self.train = true
   self.noise = torch.Tensor()
end

function SpatialDropoutX21:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.train then
      if input:dim() == 4 then
        self.noise:resize(input:size(1), input:size(2), 1, 1)
      elseif input:dim() == 3 then
        self.noise:resize(input:size(1), 1, 1)
      else
        error('Input must be 4D (nbatch, nfeat, h, w) or 3D (nfeat, h, w)')
      end
      self.noise:bernoulli(1-self.p)
      -- We expand the random dropouts to the entire feature map because the
      -- features are likely correlated accross the map and so the dropout
      -- should also be correlated.
      self.output:cmul(torch.expandAs(self.noise, input))
      self.output:div(1-self.p)
   end
   return self.output
end

function SpatialDropoutX21:updateGradInput(input, gradOutput)
   if self.train then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      self.gradInput:cmul(torch.expandAs(self.noise, input)) -- simply mask the gradients with the noise vector
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end

function SpatialDropoutX21:setp(p)
   self.p = p
end

function SpatialDropoutX21:__tostring__()
  return string.format('%s(%f)', torch.type(self), self.p)
end

function SpatialDropoutX21:clearState()
  if self.noise then
    self.noise:set()
  end
  return Parent.clearState(self)
end
