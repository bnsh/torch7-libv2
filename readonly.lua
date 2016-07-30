#!  /usr/local/torch/install/bin/th

-- This will take as input a nn.Module (Really, it just takes any Table)
-- and modifies it's updateParameters(lr) to just not do anything.

function readonly(m)
	function m:updateParameters(lr)
	end
	return m
end
