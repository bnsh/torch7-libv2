#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
#include <luaT.h>
}
#include <TH.h>
#include <THC.h>
#include <torch/utils.h>
#include <nppi.h>

extern "C" {
	int luaopen_imageutils(lua_State *L);
}

static inline void luax21_assert(lua_State *L, int predicate, const char *fmt=NULL, ...) __attribute__ ((format (printf, 3, 4)));
static inline void luax21_assert(lua_State *L, int predicate, const char *fmt, ...) {
	if (!predicate) {
		if (fmt) {
			va_list argp;
			va_start(argp, fmt);
			char buffer[4096];
			vsnprintf(buffer, 4096, fmt, argp);
			va_end(argp);
			luaL_error(L, buffer);
		}
		else {
			luaL_error(L, lua_tostring(L, -1));
		}
	}
}

/*
 * We will assume that the data we get is _already_ "channel packed" meaning the data
 * should be in this order:
 *	Y,X,C
 * as in, in Lua, we'd get Y, X, C as tensor[{Y,X,C}], as opposed to how natively
 * we'd probably get tensor[{C, Y, X}]
 * So, before we pass to any of these functions we should run
 * 	tensor:permute(2,3,1)
 */

static int gaussian_blur(lua_State *L) {
	int rv = 0;
	const THCudaTensor *kernel = (const THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");
	const THCudaTensor *img = (const THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	luax21_assert(L, kernel->nDimension == 2, "gaussian_blur demands a kernel of dimension 2!");
	luax21_assert(L, kernel->size[0] == kernel->size[1], "gaussian_blur demands a square kernel!");
	luax21_assert(L, kernel->size[0] % 2 == 1, "gaussian_blur demands an odd sized square kernel!");
	luax21_assert(L, img->nDimension == 3, "gaussian_blur demands an img of dimension 3!");
	// luax21_assert(L, img->size[0] == img->size[1], "gaussian_blur demands an square image!");

	int halfkw = (kernel->size[0]-1) / 2;
	long int y = img->size[0];
	long int x = img->size[1];
	long int c = img->size[2];

	lua_getglobal(L, "torch");
	lua_getfield(L, -1, "CudaTensor");
	lua_pushnumber(L, y - 2 * halfkw);
	lua_pushnumber(L, x - 2 * halfkw);
	lua_pushnumber(L, c);
	luax21_assert(L, lua_pcall(L, 3, 1, 0) == 0);

	// THCState *thcstate = cutorch_getstate(L);
	THCudaTensor *blurred = (THCudaTensor *)luaT_checkudata(L, -1, "torch.CudaTensor");

	NppiSize roi; roi.width = x - 2 * halfkw; roi.height = y - 2 * halfkw;
	NppiSize kernelsize; kernelsize.width = kernel->size[0]; kernelsize.height = kernel->size[1];
	NppiPoint anchor; anchor.x = halfkw; anchor.y = halfkw;

// nppiFilter(cudaGorilla + 5775, 4608, cudaBlurred, 4488, {374,374}, cudaKernel, {11,11}, {5,5})

// fprintf(stderr, "nppiFilter(cudaGorilla + %lu, %lu, cudaBlurred, %lu, {%d,%d}, cudaKernel, {%d,%d}, {%d,%d})\n", (halfkw*img->stride[0]+halfkw*img->stride[1]), sizeof((*img->storage->data))*img->stride[0], sizeof((*img->storage->data))*blurred->stride[0], roi.width, roi.height, kernelsize.width, kernelsize.height, anchor.x, anchor.y);
 
	int ss = nppiFilter_32f_C3R(
		img->storage->data + (halfkw * img->stride[0] + halfkw * img->stride[1]),
		sizeof((*img->storage->data))*img->stride[0],
		blurred->storage->data,
		sizeof((*blurred->storage->data))*blurred->stride[0],
		roi,
		kernel->storage->data,
		kernelsize,
		anchor
	);
	luax21_assert(L, ss == 0, "nppiFilter_32f_C3R returned %d", ss);

	rv = 1;
	return rv;
}

/* rotatecrop takes a theta and a _square_ image,
 * rotates the image by that theta, and crops out the
 * _largest_ square that would fit within there. That
 * square will be of size floor(sz / (sin(theta) + cos(theta)))
 */
static int rotatecrop(lua_State *L) {
	int rv = 0;
	double theta = lua_checknumber(L, 1);
	const THCudaTensor *img = (const THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	luax21_assert(L, img->nDimension == 3, "rotatecrop demands an img of dimension 3!");

	int halfkw = (kernel->size[0]-1) / 2;
	long int y = img->size[0];
	long int x = img->size[1];
	long int c = img->size[2];

	lua_getglobal(L, "torch");
	lua_getfield(L, -1, "CudaTensor");
	lua_pushnumber(L, y - 2 * halfkw);
	lua_pushnumber(L, x - 2 * halfkw);
	lua_pushnumber(L, c);
	luax21_assert(L, lua_pcall(L, 3, 1, 0) == 0);

}

static const struct luaL_reg imageutils[] = {
	{"gaussian_blur", gaussian_blur },
	{"rotatecrop", rotatecrop },
	{NULL, NULL}
};

int luaopen_imageutils(lua_State *L) {
	luaL_openlib(L, "imageutils", imageutils, 0);
	return(1);
}
