#! /usr/local/torch/install/bin/th

-- This is adapted from http://www.johndcook.com/blog/normal_cdf_inverse/

function inverse_cdf(realp)
	local function rational_approximation(t)
		local c = torch.DoubleTensor({{2.515517, 0.802853, 0.010328}}):t():typeAs(t)
		local d = torch.DoubleTensor({{1.432788, 0.189269, 0.001308}}):t():typeAs(t)
		local tv = torch.cat(torch.cat(torch.pow(t,0), t, 2), torch.pow(t,2),2)
		local ctv = torch.mm(tv, c)
		local dtv = torch.cmul(torch.mm(tv, d), t) + 1
		local rv = torch.csub(t, torch.cdiv(ctv, dtv))

		return rv
	end

-- We want to give p as a matrix. But.. This is problematic, because of the matrix
-- multiplication above. But, it's no problem. We can recast it as a vector ((row*col), 1)
-- do our computation, then recast the result back as a matrix, right? easy peasy.
	local p = realp:view(realp:numel(), 1)
	assert(p:le(1):all())
	assert(p:ge(0):all())

	local oneiflt = p:lt(0.5):long():mul(2):add(-1):typeAs(p)
	local oneifge = -oneiflt

	local stage = 1
	local rv = torch.cmul(oneifge, rational_approximation(torch.sqrt(torch.mul(torch.log(torch.add(torch.cmul(p, oneiflt), p:ge(0.5):typeAs(p))),-2)))):reshape(realp:size())
	return rv
end


