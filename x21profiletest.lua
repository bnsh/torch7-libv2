#! /usr/local/torch/install/bin/th

require "x21profile"

function realfactorial(x)
	if x == 0 then return 1 else return x * realfactorial(x-1) end
end

function factorial(x)
	return realfactorial(x)
end

x21profile.start("/dev/tty")
factorial(4)
x21profile.stop()

factorial(12)
