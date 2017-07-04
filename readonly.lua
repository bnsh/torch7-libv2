#!  /usr/local/torch/install/bin/th

require "fprintf"
-- This will take as input a nn.Module (Really, it just takes any Table)
-- and modifies it's updateParameters(lr) to just not do anything.

function readonly(m)
	m:apply(function (s)
		s.pfuncwritable = {
			updateParameters=s.updateParameters,
			parameters=s.parameters
		}
		s.updateParameters = function (lr) end
		s.parameters = function() return nil end
	end)
	return m
end

function writable(m)
	m:apply(function (s)
		if s.pfuncwritable ~= nil then
			s.updateParameters = s.pfuncwritable.updateParameters
			s.parameters = s.pfuncwritable.parameters
			s.pfuncwritable = nil
		end
	end)
	return m
end
