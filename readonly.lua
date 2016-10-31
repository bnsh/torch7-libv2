#!  /usr/local/torch/install/bin/th

require "fprintf"
-- This will take as input a nn.Module (Really, it just takes any Table)
-- and modifies it's updateParameters(lr) to just not do anything.

function readonly(m)
	m:apply(function (s)
		s.updateParameters = function (lr) end
		s.parameters = function() return nil end
	end)
	return m
end
