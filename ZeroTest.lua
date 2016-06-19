require "nn"
require "Zero"

local jac = nn.Jacobian
print(jac.testJacobian(nn.Zero(5,3,17), torch.rand(32,32)))
