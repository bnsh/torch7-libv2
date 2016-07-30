#include <stdio.h>
#include <string.h>
#include <assert.h>
extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
#include <luaT.h>
}
#include <TH.h>
#include <THC.h>

extern "C" {
	int luaopen_normalize(lua_State *L);
}

/*
 * OK, so we want a function that takes as input a DoubleTensor, and will
 * _modify_ the _same_ tensor so that it is normalized. Optionally,
 * it can also take a regularization parameter, which will be
 * added to all items of the tensor before regularizing.
 */

static int lua_normalize(lua_State *L) {
	THDoubleTensor *t = (THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
	if ((lua_gettop(L)) < 2) lua_pushnumber(L, 0.0); // If it's not there, pop a numeric there, so we can use it later.
	double reg = lua_tonumber(L, 2);
	if (t->nDimension != 2) luaL_error(L, "Dimensions of tensor must be 2 not %d.", t->nDimension);
	else {
		unsigned long int rows = t->size[0];
		unsigned long int columns = t->size[1];

		for (unsigned long int i = 0; i < rows; ++i) {
			double total = reg * columns;
			for (unsigned long int j = 0; j < columns; ++j) total += t->storage->data[t->storageOffset+i*t->stride[0]+j];
			for (unsigned long int j = 0; j < columns; ++j) {
				double q = 1.0 / columns; // This will _only_ happen if reg is 0 _and_ total is zero (of course, total == 0 => reg == 0.
				if (total != 0.0) 
					q = (reg + t->storage->data[t->storageOffset+i*t->stride[0]+j]) / total;
				t->storage->data[t->storageOffset+i*t->stride[0]+j] = q;
				
			}
		}
	}

	return 0;
}

static const struct luaL_reg normalize[] = {
	{"normalize", lua_normalize },
	{NULL, NULL}
};

int luaopen_normalize(lua_State *L) {
	luaL_openlib(L, "normalize", normalize, 0);
	lua_getfield(L, -1, "normalize");
	lua_setglobal(L, "normalize");
	return(0);
}
