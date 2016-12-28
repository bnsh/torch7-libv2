#! /usr/local/torch/install/bin/th

require "cephes"
require "inverse_cdf"

function main()
	local iters = 65536
	local data = torch.DoubleTensor(iters, 1):uniform(0,1)
	local zscores = inverse_cdf(data)
	local recomputed = (1 + cephes.erf(zscores / math.sqrt(2))) / 2
	local err = (recomputed-data)
	local err2 = torch.pow(err, 2)
	local sumerr2 = err2:sum()
	local rms = torch.sqrt(sumerr2 / iters)
	print(rms)
	assert(rms <= 1e-4)
end

main()
