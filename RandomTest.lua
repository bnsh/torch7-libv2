require "nn"
require "cudnn"
require "Random"

--[=[
Because Jacobian outputs (f(x+e)-f(x-e))/2e, and f(x+e) and f(x-e) are both random variables,
This "error" will never be 0. What will it in fact be?

I guess it's Integrate[(x1-x2)PDF[NormalDistribution[m,sd],x1]PDF[NormalDistribution[m,sd],x2],{x1,-Infinity,Infinity},{x2,-Infinity,Infinity},Assumptions->{sd>=0}] ?
But, this is 0.
]=]
local jac = nn.Jacobian
print(jac.testJacobian(nn.Random({5,3,17},0,1), torch.rand(32,32)))
