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
	int luaopen_tfidf(lua_State *L);
}

// Make _all_ allocation happen explicitly in lua.

/*
 * So, idf takes as input 2 DoubleTensors and _possibly_ a regularization term.
 * if the regularization term is not specified, it is taken to be 1.
 * Basically, we're computing log(N+reg*terms) - log(docswithtermt+1)
 */

static int lua_idf(lua_State *L) {
	const THDoubleTensor *t = (const THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
	THDoubleTensor *idf = (THDoubleTensor *)luaT_checkudata(L, 2, "torch.DoubleTensor");
	if (lua_gettop(L) < 3) lua_pushnumber(L, 0.0);
	double reg = lua_tonumber(L, 3);

	if (t->nDimension != 2) luaL_error(L, "Dimension of tensor must be 2 not %d.", t->nDimension);
	else if ((t->size[1] != idf->size[1]) || (idf->size[0] != 1)) luaL_error(L, "idf for a rawcounts of size %dx%d should be 1x%d, not %dx%d", t->size[0], t->size[1], t->size[1], idf->size[0], idf->size[1]);
	else {
		for (int j = 0; j < t->size[1]; ++j) {
			idf->storage->data[idf->storageOffset+0*idf->stride[0]+j] = log(t->size[0]+t->size[1]*reg);
			double nt = reg;
			for (int i = 0; i < t->size[0]; ++i) {
				if (t->storage->data[t->storageOffset+i*t->stride[0]+j] >= 1.0) nt += 1.0;
			}
			idf->storage->data[idf->storageOffset+0*idf->stride[0]+j] -= log(nt);
		}
	}
	return 0;
}

/*
 * So, tfidf should take as input 2 DoubleTensors, and possibly a regularization term
 * the second parameter, if it exists, will be the idf to be used.
 * if not, the idf will have to be computed.
 * the third parameter is the "regularization"
 * So, we're going to do (frequency(term in doc d)+reg)/(#{total terms in doc d} + reg*terms)
 */

static int lua_tfidf(lua_State *L) {
	THDoubleTensor *t = (THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
	const THDoubleTensor *idf = (const THDoubleTensor *)luaT_checkudata(L, 2, "torch.DoubleTensor");
	if ((lua_gettop(L)) < 3) lua_pushnumber(L, 0);
	double reg = lua_tonumber(L, 3);

	if (t->nDimension != 2) luaL_error(L, "Dimension of tensor must be 2 not %d.", t->nDimension);
	else if ((t->size[1] != idf->size[1]) || (idf->size[0] != 1)) luaL_error(L, "idf for a rawcounts of size %dx%d should be 1x%d, not %dx%d", t->size[0], t->size[1], t->size[1], idf->size[0], idf->size[1]);
	else {
		unsigned long int rows = t->size[0];
		unsigned long int columns = t->size[1];

		for (unsigned long int i = 0; i < rows; ++i) {
			double denominator = reg * columns;
			for (unsigned long int j = 0; j < columns; ++j)
				denominator = denominator + t->storage->data[t->storageOffset+i*t->stride[0]+j];

			for (unsigned long int j = 0; j < columns; ++j) {
				double numerator = reg + t->storage->data[t->storageOffset+i*t->stride[0]+j];
				double ouridf = idf->storage->data[idf->storageOffset+0*idf->stride[0]+j];
				t->storage->data[t->storageOffset+i*t->stride[0]+j] = (numerator / denominator) * ouridf;
			}
		}
	}


	return 0;
}

static const struct luaL_reg tfidf[] = {
	{"tfidf", lua_tfidf },
	{"idf", lua_idf },
	{NULL, NULL}
};

int luaopen_tfidf(lua_State *L) {
	luaL_openlib(L, "tfidf", tfidf, 0);
	lua_setglobal(L, "tfidf");
	return(0);
}
