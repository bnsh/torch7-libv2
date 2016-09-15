#! /usr/local/torch/install/bin/th

function kldivergence(output, target)
--[=[
This should compute _exactly_ like nn.DistKLDivCriterion. But, then
why should this even exist?? Because often, I need to know the output
for a particular index, and DistKLDivCriterion offers no way of getting
the error for a _particular_ value.
We assume that output is designed to go into DistKLDivCriterion, so
it should already be the log probability.
KLDiv = Sum(target * (output - log(target)))
Problems arise when target==0, of course.
]=]
	local odds = torch.log(target) - output
	local matrix = torch.cmul(target, odds)
	local matrixv = matrix:view(-1)
-- CudaByteTensor doesn't have "nonzero()" *sigh*
	local zeros = target:view(-1):lt(1e-10):eq(1):byte():nonzero():squeeze()
	if torch.type(zeros) == 'number' then
		zeros = torch.LongTensor({zeros})
	end
	if zeros:numel() > 0 then
		matrixv:indexFill(1,zeros,0)
	end
	local rv = matrix:sum(2)
	return rv
end
