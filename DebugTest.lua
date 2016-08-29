require "nn"
require "Debug"

local jac = nn.Jacobian
print(jac.testJacobian(nn.Debug(5,3,17), torch.rand(32,32)))
