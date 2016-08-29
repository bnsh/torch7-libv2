require "nn"
require "cudnn"
require "Random"

local jac = nn.Jacobian
print(jac.testJacobian(nn.Random(5,3,17), torch.rand(32,32)))
