#! /usr/local/torch/install/bin/th

require "cephes"
require "inverse_cdf"

function main()
	local iters = 65536
	local m = 64
	local data = torch.DoubleTensor(iters, m):uniform(0,1)
	local zscores = inverse_cdf(data)
	print(zscores:size())
	local recomputed = (1 + cephes.erf(zscores / math.sqrt(2))) / 2
	local err = (recomputed-data)
	local err2 = torch.pow(err, 2)
	local sumerr2 = err2:sum()
	local rms = torch.sqrt(sumerr2 / data:numel())
	print(rms)
	assert(rms <= 1e-4)
	local p = torch.DoubleTensor({0.0, 1.0})
	local z = inverse_cdf(p)
	print(p)
	print(z)
	p = torch.DoubleTensor({0.1, 0.9})
	z = inverse_cdf(p)
	print(p)
	print(z)
	p = torch.DoubleTensor({0.1, 1.0})
	z = inverse_cdf(p)
	print(p)
	print(z)
	p = torch.DoubleTensor({0.0, 0.9})
	z = inverse_cdf(p)
	print(p)
	print(z)
end

main()
