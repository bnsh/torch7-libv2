#!/usr/local/torch/install/bin/th

function no_globals(main)
	local before = { }
	for k, v in pairs(_G)
	do
		before[k] = true
	end
	local function verify()
		local problems = 0
		for k, v in pairs(_G)
		do
			if before[k] == nil
			then
				io.stderr:write(string.format("GLOBAL LEAK! WTF IS '%s'?! LITTER BUG!\n", k))
				problems = problems + 1
			end
		end
		return problems
	end
	main(arg, verify)
	assert(0 == verify())
end
