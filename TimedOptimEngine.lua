#! /usr/local/torch/install/bin/th


local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'
local doc = require 'argcheck.doc'
local json = require "dkjson"
require "timehires"

doc[[
### tnt.TimedOptimEngine

The `TimedOptimEngine` is really just a subclass of OptimEngine that offers

	* getTimingInfo() which will return a table with keys
		for each hook, and values like { count=callcounts, elapsed=timespent }

	* resetTimingInfo() which will reset the timer information.

	* dumpTimingInfo(fn) which will dump the timinginfo to fn in json format.
]]

local TimedOptimEngine, OptimEngine = torch.class('tnt.TimedOptimEngine', 'tnt.OptimEngine', tnt)

TimedOptimEngine.__init = argcheck{
	{name="self", type="tnt.TimedOptimEngine"},
	call = function(self)
		OptimEngine.__init(self)
		self.timing_info = { }
		self.lastmark = timehires.timehires()
-- We need to change the __call part of self.hooks from Engine
		local hookmt = getmetatable(self.hooks)
		local oldcall = hookmt.__call
		hookmt.__call = function(hooks, name, ...)
			self:mark('before' .. name)
			local rv = oldcall(hooks, name, ...)
			self:mark(name)
			return rv
		end
		setmetatable(self.hooks, hookmt)
	end
}

TimedOptimEngine.mark = argcheck{
	{name="self", type="tnt.TimedOptimEngine"},
	{name="name", type="string"},
	call = function(self, name)
		if self.timing_info[name] == nil then
			self.timing_info[name] = {
				count=0,
				elapsed=0
			}
		end
		self.timing_info[name].count = self.timing_info[name].count + 1
		local now = timehires.timehires()
		self.timing_info[name].elapsed = self.timing_info[name].elapsed + (now - self.lastmark)
		self.lastmark = now
	end
}

TimedOptimEngine.getTimingInfo = argcheck{
	{name="self", type="tnt.TimedOptimEngine"},
	call = function(self) return self.timing_info end
}


TimedOptimEngine.resetTimingInfo = argcheck{
	{name="self", type="tnt.TimedOptimEngine"},
	call = function(self) self.timing_info = { } end
}

TimedOptimEngine.dumpTimingInfo = argcheck{
	{name="self", type="tnt.TimedOptimEngine"},
	{name="filename", type="string"},
	call = function(self, filename)
		local fh = io.open(filename, "w")
		if fh ~= nil then
			fh:write(json.encode(self.timing_info, { indent=true }))
			fh:close()
		end
	end
}
