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
#include <npp.h>
#include <nppi.h>

extern "C" {
	int luaopen_imageutils(lua_State *L);
}

// static NppiInterpolationMode interpolationmode = NPPI_INTER_LINEAR;
static NppiInterpolationMode interpolationmode = NPPI_INTER_CUBIC2P_B05C03;

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
	luax21_assert(L, img->size[2] == 3, "gaussian_blur demands a color image with no alpha!");

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
	double theta = luaL_checknumber(L, 1);
	const THCudaTensor *img = (const THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	luax21_assert(L, img->nDimension == 3, "rotatecrop demands an img of dimension 3!");
	luax21_assert(L, img->size[0] == img->size[1], "rotatecrop demands an square image!");
	luax21_assert(L, img->size[2] == 3, "rotatecrop demands a color image with no alpha!");

	long int sz = img->size[0];
	long int c = img->size[2];

	// So, if we've rotated an image of size sz _squared_ then
	// our output should be of size sz / (Sin[t]+Cos[t])
	// But, we don't care if it's above 90 degrees, (in which case, it's
	// just a corner shift, etc. etc. etc.) and at the end, we come up with
	// sz / (Sqrt[2] Cos[t - Pi/4 - Floor[(2 t / Pi)] Pi / 2])

	double denominator = sqrt(2.0) * cos(theta - M_PI / 4.0 - floor(2 * theta / M_PI) * M_PI / 2.0);
	int rotatedsz = floor(sz / denominator);

	lua_getglobal(L, "torch");
	lua_getfield(L, -1, "CudaTensor");
	lua_pushnumber(L, rotatedsz);
	lua_pushnumber(L, rotatedsz);
	lua_pushnumber(L, c);
	luax21_assert(L, lua_pcall(L, 3, 1, 0) == 0);

	lua_getfield(L, -1, "zero");
	lua_pushvalue(L, -2);
	luax21_assert(L, lua_pcall(L, 1, 0, 0) == 0);

	NppiSize imgsz; imgsz.width = sz; imgsz.height = sz;
	NppiRect imgroi; imgroi.x = 0; imgroi.y = 0; imgroi.width = sz; imgroi.height = sz;
// So, what we want here, is the _center_ of the image.
// (sz - rotatedsz) / 2 to (sz + rotatedsz / 2)
	NppiRect rotatedroi; rotatedroi.x = 0; rotatedroi.y = 0; rotatedroi.width = rotatedsz; rotatedroi.height = rotatedsz;
	THCudaTensor *rotated = (THCudaTensor *)luaT_checkudata(L, -1, "torch.CudaTensor");

// This was laboriously calculated.
	double shiftx =  (sz - sz * cos(theta) - sz * sin(theta)) / 2. - (sz - rotatedsz)/2.;
	double shifty =  (sz + sz * sin(theta) - sz * cos(theta)) / 2. - (sz - rotatedsz)/2.;

	int ss = nppiRotate_32f_C3R(
		img->storage->data,
		imgsz,
		sizeof((*img->storage->data))*img->stride[0],
		imgroi,
		rotated->storage->data,
		sizeof((*rotated->storage->data))*rotated->stride[0],
		rotatedroi,
		theta * 180.0 / M_PI,
		shiftx,
		shifty,
		interpolationmode
	);
	luax21_assert(L, ss >= 0, "nppiFilter_32f_C3R returned %d", ss);

	rv = 1;
	return rv;
}

/* scale takes 2 numbers and an image, and scales
 * the image to fit those numbers. Aspect ratio is _NOT_ preserved.
 */
static int scale(lua_State *L) {
	int rv = 0;
	double scaledx = luaL_checknumber(L, 1);
	double scaledy = luaL_checknumber(L, 2);
	const THCudaTensor *img = (const THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
	luax21_assert(L, img->nDimension == 3, "scale demands an img of dimension 3!");
	luax21_assert(L, img->size[2] == 3, "scale demands a color image with no alpha!");

	long int sy = img->size[0];
	long int sx = img->size[1];
	long int c = img->size[2];

	lua_getglobal(L, "torch");
	lua_getfield(L, -1, "CudaTensor");
	lua_pushnumber(L, scaledy);
	lua_pushnumber(L, scaledx);
	lua_pushnumber(L, c);
	luax21_assert(L, lua_pcall(L, 3, 1, 0) == 0);

	lua_getfield(L, -1, "zero");
	lua_pushvalue(L, -2);
	luax21_assert(L, lua_pcall(L, 1, 0, 0) == 0);

	NppiSize imgsz; imgsz.width = sx; imgsz.height = sy;
	NppiRect imgroi; imgroi.x = 0; imgroi.y = 0; imgroi.width = sx; imgroi.height = sy;
	NppiRect scaledroi; scaledroi.x = 0; scaledroi.y = 0; scaledroi.width = scaledx; scaledroi.height = scaledy;
	THCudaTensor *scaled = (THCudaTensor *)luaT_checkudata(L, -1, "torch.CudaTensor");
	int ss = nppiResizeSqrPixel_32f_C3R(
		img->storage->data,
		imgsz,
		sizeof((*img->storage->data))*img->stride[0],
		imgroi,
		scaled->storage->data,
		sizeof((*scaled->storage->data))*scaled->stride[0],
		scaledroi,
		scaledx/sx,
		scaledy/sy,
		0,
		0,
		interpolationmode
	);
	luax21_assert(L, ss >= 0, "nppiResizeSqrPixel_32f_C3R returned %d", ss);

	rv = 1;
	return rv;
}

/* normalized_square takes a size and an image, and 
 * crops out the center portion of the image scaled to be that size
 */
static int normalized_square(lua_State *L) {
	int rv = 0;
	double sz = luaL_checknumber(L, 1);
	const THCudaTensor *img = (const THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	luax21_assert(L, img->nDimension == 3, "normalized_square demands an img of dimension 3!");
	luax21_assert(L, img->size[2] == 3, "normalized_square demands a color image with no alpha!");

	long int sy = img->size[0];
	long int sx = img->size[1];
	long int c = img->size[2];


	lua_getglobal(L, "torch");
	lua_getfield(L, -1, "CudaTensor");
	lua_pushnumber(L, sz);
	lua_pushnumber(L, sz);
	lua_pushnumber(L, c);
	luax21_assert(L, lua_pcall(L, 3, 1, 0) == 0);

	lua_getfield(L, -1, "zero");
	lua_pushvalue(L, -2);
	luax21_assert(L, lua_pcall(L, 1, 0, 0) == 0);

	NppiSize imgsz; imgsz.width = sx; imgsz.height = sy;
// Our region of interest is the center.
	NppiRect imgroi; imgroi.x = 0; imgroi.y = 0; imgroi.width = sx; imgroi.height = sy;
	double scalefactor = 1.0;
	double xshift = 0.0;
	double yshift = 0.0;
	// double yshift = 0.0;
	if (sx < sy) {
		scalefactor = sz / sx;
		xshift = 0;
		yshift = (sx - sy) / 2.0;
	}
	else {
		scalefactor = sz / sy;
		xshift = (sy - sx) / 2.0;
		yshift = 0;
	}
	NppiRect scaledroi;
	scaledroi.x = 0;
	scaledroi.y = 0;
	scaledroi.width = sz;
	scaledroi.height = sz;

	THCudaTensor *scaled = (THCudaTensor *)luaT_checkudata(L, -1, "torch.CudaTensor");
	int ss = nppiResizeSqrPixel_32f_C3R(
		img->storage->data,
		imgsz,
		sizeof((*img->storage->data))*img->stride[0],
		imgroi,
		scaled->storage->data,
		sizeof((*scaled->storage->data))*scaled->stride[0],
		scaledroi,
		scalefactor,
		scalefactor,
		xshift,
		yshift,
		interpolationmode
	);
	luax21_assert(L, ss >= 0, "nppiResizeSqrPixel_32f_C3R returned %d", ss);

	rv = 1;
	return rv;
}

/* horizontalflip takes an image, and horizontally flips it.
 */
static int horizontalflip(lua_State *L) {
	int rv = 0;
	const THCudaTensor *img = (const THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");
	luax21_assert(L, img->nDimension == 3, "horizontalflip demands an img of dimension 3!");
	luax21_assert(L, img->size[2] == 3, "horizontalflip demands a color image with no alpha!");

	long int sy = img->size[0];
	long int sx = img->size[1];

	NppiSize imgroi; imgroi.width = sx; imgroi.height = sy;
	int ss = nppiMirror_32f_C3IR(
		img->storage->data,
		sizeof((*img->storage->data))*img->stride[0],
		imgroi,
		NPP_VERTICAL_AXIS
	);
	luax21_assert(L, ss >= 0, "nppiMirror_32f_C3IR returned %d", ss);

	rv = 1;
	return rv;
}

/*
	colorwash takes a number p and an img. It "enhances" or washes out the color depending on
	the value of p.
	p > 1 is kind of enhancing color. But, this probably should be the domain of contrast
	p = 1 means full color
	p = 0 means grey scale
	p < 0 will negate images.

	R = p r + (1-p)(r + g + b)/3
	R = p r + ((r + g + b) - pr - pg - pb)/3
	R = p r + (r + g + b - pr - pg - pb) / 3
	R = (3 pr + r + g + b - pr - pg - pb) / 3
	R = (2 pr + r + g + b - pg - pb) / 3
	R = (r (2 p + 1) + g (1-p) + b (1-p)) / 3
	R = ([(2p+1) (1-p) (1-p)] Transpose[r g b]) / 3

	Is this right?
		p = 0 means
			[1 1 1] [r g b] / 3 check.
		p = 1 means [3 0 0] [r g b] / 3 check.

	So, more generally, we need to do this:
		[ (2p+1),  (1-p),  (1-p) ] [ r ] = (2p+1)r +  (1-p)g +  (1-p)b
		[  (1-p), (2p+1),  (1-p) ] [ g ] =  (1-p)r + (2p+1)g +  (1-p)b
		[  (1-p),  (1-p), (2p+1) ] [ b ] =  (1-p)r +  (1-p)g + (2p+1)b

	So, let"s take an identity matrix.

	a 0 + b == 1-p
	a 1 + b == 1+2p

	a + 1-p = 1+2p
	a -p = 2p
	a = 3p?

	So, torch.eye(3):mul(3p):add(1-p) ?
	pixels is of size 3 x {width*height}
*/
static int colorwash(lua_State *L) {
	int rv = 0;
	double p = luaL_checknumber(L, 1);
	const THCudaTensor *img = (const THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	luax21_assert(L, img->nDimension == 3, "colorwash demands an img of dimension 3!");
	luax21_assert(L, img->size[2] == 3, "colorwash demands a color image with no alpha!");

	long int sy = img->size[0];
	long int sx = img->size[1];

	float twist[3][4];
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 4; ++j) {
			twist[i][j] = 0.0;
			if (i == j) twist[i][j] = (2.0 * p + 1.0) / 3.;
			else twist[i][j] = (1.0 - p) / 3.;
		}
	}

	NppiSize imgroi; imgroi.width = sx; imgroi.height = sy;
	int ss = nppiColorTwist_32f_C3IR(
		img->storage->data,
		sizeof((*img->storage->data))*img->stride[0],
		imgroi,
		twist
	);
	luax21_assert(L, ss >= 0, "nppiColorTwist_32f_C3IR returned %d", ss);

	rv = 1;
	return rv;
}

/*
	adjust_contrast takes contrast, brightness and an img. It adjust the constrast
	depending on the value of contrast and brightness.

	a contrast of 1, and a brightness of 0 returns the same image.
	a contrast of 0 will basically return the mean color of the image.
	a contrast of 2 will enhance the colors of the image. (It may oversaturate)
*/
static int adjust_contrast(lua_State *L) {
	int rv = 0;
	double cntrst = luaL_checknumber(L, 1);
	double bright = luaL_checknumber(L, 2);
	const THCudaTensor *img = (const THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
	luax21_assert(L, img->nDimension == 3, "adjust_contrast demands an img of dimension 3!");
	luax21_assert(L, img->size[2] == 3, "adjust_contrast demands a color image with no alpha!");

// So, first we have to compute the mean and the standard deviation of the image.
	double *gpumean = nppsMalloc_64f(3);
	double *gpustd = nppsMalloc_64f(3);


	long int sy = img->size[0];
	long int sx = img->size[1];
	NppiSize imgroi; imgroi.width = sx; imgroi.height = sy;

	int buffersz = -1;
	int ss = nppiMeanStdDevGetBufferHostSize_32f_C3CR(imgroi, &buffersz);
	luax21_assert(L, ss == 0, "nppiMeanStdDevGetBufferHostSize_32f_C3CR returned %d", ss);

	Npp8u *scratchspace = nppsMalloc_8u(buffersz);

	for (int i = 0; i < 3; ++i) {
		ss = nppiMean_StdDev_32f_C3CR(
			img->storage->data,
			sizeof((*img->storage->data))*img->stride[0],
			imgroi,
			(1+i),
			scratchspace,
			&gpumean[i],
			&gpustd[i]
		);
		luax21_assert(L, ss == 0, "nppiMean_StdDev_32f_C3CR returned %d", ss);
	}

	/* Ugh. Why does Mean_StdDev not do this for us?! */
	double cpumean[3]; luax21_assert(L, cudaSuccess == cudaMemcpy(cpumean, gpumean, sizeof(cpumean), cudaMemcpyDeviceToHost), "cudaMemcpy failed!");
	double cpustd[3]; luax21_assert(L, cudaSuccess == cudaMemcpy(cpustd, gpustd, sizeof(cpustd), cudaMemcpyDeviceToHost), "cudaMemcpy failed!");
	for (int i = 0; i < 3; ++i) {
// if the standard deviation is 0, then all the pixels are the same as the mean.
// if we artificially set std = 1, then
// 	(z-mean)/1 = 0 will still be true, but we'll avoid the NaN insanity.
		if (cpustd[i] == 0.0) cpustd[i] = 1.0;
	}

/* OK, this is a big mess. So, first we subtract the mean
	submean = {
		{ 1, 0, 0, -mr },
		{ 0, 1, 0, -mg },
		{ 0, 0, 1, -mb },
		{ 0, 0, 0,   1 }
	}

   Then we divide by std
	divstd = {
		{ 1/stdr, 0, 0, 0 },
		{ 0, 1/stdg, 0, 0 },
		{ 0, 0, 1/stdb, 0 },
		{ 0, 0,      0, 1 }
	}

   Then we multiply by contrast.
	contrast = {
		{ cntrst, 0, 0, 0 },
		{ 0, cntrst, 0, 0 },
		{ 0, 0, cntrst, 0 },
		{ 0, 0,      0, 1 }
	}

   Add brightness
	brightness = {
		{ 1, 0, 0, bright },
		{ 0, 1, 0, bright },
		{ 0, 0, 1, bright },
		{ 0, 0, 0,      1 }
	}

   multiply by std
	mulstd = {
		{ stdr, 0, 0, 0 },
		{ 0, stdg, 0, 0 },
		{ 0, 0, stdb, 0 },
		{ 0, 0,    0, 1 }
	}

   add mean
	addmean = {
		{ 1, 0, 0, mr },
		{ 0, 1, 0, mg },
		{ 0, 0, 1, mb },
		{ 0, 0, 0,  1 }
	}

   All together, thank you Mathematica:
	addmean . mulstd . brightness . contrast . divstd . submean = {
		{ cntrst, 0, 0, mr-cntrst * mr + bright * stdr },
		{ 0, cntrst, 0, mg-cntrst * mg + bright * stdg },
		{ 0, 0, cntrst, mb-cntrst * mb + bright * stdb },
		{ 0, 0,      0,                              1 }
	}
 */

	float twist[3][4];
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 4; ++j) {
			if (j == 4) twist[i][j] = cpumean[i] - cntrst * cpumean[i] + bright * cpustd[i];
			else if (i == j) twist[i][j] = cntrst;
			else twist[i][j] = 0;
		}
	}

	ss = nppiColorTwist_32f_C3IR(
		img->storage->data,
		sizeof((*img->storage->data))*img->stride[0],
		imgroi,
		twist
	);
	luax21_assert(L, ss >= 0, "nppiColorTwist_32f_C3IR returned %d", ss);
	nppsFree(gpumean); gpumean = NULL;
	nppsFree(gpustd); gpustd = NULL;
	nppsFree(scratchspace); scratchspace = NULL;
	rv = 1;
	return rv;
}

static const struct luaL_reg imageutils[] = {
	{"gaussian_blur", gaussian_blur },
	{"rotatecrop", rotatecrop },
	{"scale", scale },
	{"horizontalflip", horizontalflip },
	{"colorwash", colorwash },
	{"adjust_contrast", adjust_contrast },
	{"normalized_square", normalized_square },
	{NULL, NULL}
};

int luaopen_imageutils(lua_State *L) {
	luaL_openlib(L, "imageutils", imageutils, 0);
	return(1);
}
