#include "pipeline_impl.H"
extern "C" {
#include <lauxlib.h>
}
#include <torch/utils.h>

static NppiInterpolationMode interpolationmode = NPPI_INTER_CUBIC;

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

static inline double lerp(double percent, double min, double max) {
	return min + percent * (max-min);
}

pipeline_impl::pipeline_impl() {
	_scratch_floats[0] = NULL;
	_scratch_floats[1] = NULL;
}

pipeline_impl::~pipeline_impl() {
	for (int i = 0; i < 2; ++i) {
		if (_scratch_floats[i] != NULL) delete _scratch_floats[i]; _scratch_floats[i] = NULL;
	}
}

void pipeline_impl::grab_temporaries(const NppiSize& sz) {
	int szf = sz.width * sz.height * 3;
	for (int i = 0; i < 2; ++i) {
		if ((_scratch_floats[i] == NULL) || (_scratch_floats[i]->size() < szf)) {
			if (_scratch_floats[i] != NULL) delete _scratch_floats[i]; _scratch_floats[i] = NULL;
			_scratch_floats[i] = new scratch<Npp32f>(szf);
		}
	}
}

void pipeline_impl::center_filter(lua_State *L, const Npp32f *input, const NppiSize& inputsz, int inputlinesz, Npp32f *output, const NppiSize& outputsz, int outputlinesz) {
	grab_temporaries(inputsz);
	grab_temporaries(outputsz);

	NppiSize currsz = inputsz;
	int currlinesz = inputlinesz;
// scale
	normalized_square.scaledx = outputsz.width * 11 / 10;
	normalized_square.scaledy = outputsz.height * 11 / 10;
	normalized_square(L, input, currsz, currlinesz, _scratch_floats[1]->space(), currsz, currlinesz); swap_scratch(_scratch_floats);

	center_crop.szx = outputsz.width;
	center_crop.szy = outputsz.height;
	center_crop(L, _scratch_floats[0]->space(), currsz, currlinesz, output, currsz, currlinesz); swap_scratch(_scratch_floats);

// Now, verify that currsz and outputsz are the same and that outputlinesz and currlinesz are the same.
	luax21_assert(L, (outputsz.width == currsz.width), "The outputsz.width did not match the currsz.width! Odds are there's a memory problem imminent.");
	luax21_assert(L, (outputsz.height == currsz.height), "The outputsz.height did not match the currsz.height! Odds are there's a memory problem imminent.");
	luax21_assert(L, (outputlinesz == currlinesz), "The outputlinesz did not match the currlinesz! Odds are there's a memory problem imminent.");
}

void pipeline_impl::random_filter(lua_State *L, const Npp32f *input, const NppiSize& inputsz, int inputlinesz, Npp32f *output, const NppiSize& outputsz, int outputlinesz) {
	luax21_assert(L, (outputsz.width == outputsz.height), "random_filter expects output to be square.");
	grab_temporaries(inputsz);
	grab_temporaries(outputsz);

	NppiSize currsz = inputsz;
	int currlinesz = inputlinesz;

	noop(L, input, currsz, currlinesz, _scratch_floats[1]->space(), currsz, currlinesz); swap_scratch(_scratch_floats);

	int mode = 0xff;

// random_square_crop 01
	random_square_crop.minszx = (int)(ceil(outputsz.width * sqrt(2.0)));
	random_square_crop.minszy = (int)(ceil(outputsz.height * sqrt(2.0)));
	if (mode & 0x01) { random_square_crop(L, _scratch_floats[0]->space(), currsz, currlinesz, _scratch_floats[1]->space(), currsz, currlinesz); swap_scratch(_scratch_floats); }

// gaussian_blur 02
	if (mode & 0x02) { gaussian_blur(L, _scratch_floats[0]->space(), currsz, currlinesz, _scratch_floats[1]->space(), currsz, currlinesz); swap_scratch(_scratch_floats); }

// rotatecrop 04
	if (mode & 0x04) { rotatecrop(L, _scratch_floats[0]->space(), currsz, currlinesz, _scratch_floats[1]->space(), currsz, currlinesz); swap_scratch(_scratch_floats); }

// Actually. scaling is compulsory.
	scale.scaledx = outputsz.width;
	scale.scaledy = outputsz.height;
	scale(L, _scratch_floats[0]->space(), currsz, currlinesz, _scratch_floats[1]->space(), currsz, currlinesz); swap_scratch(_scratch_floats);

// horizontalflip 08
	horizontalflip.szx = outputsz.width;
	horizontalflip.szy = outputsz.height;
	if (mode & 0x08) { horizontalflip(L, _scratch_floats[0]->space(), currsz, currlinesz, _scratch_floats[1]->space(), currsz, currlinesz); swap_scratch(_scratch_floats); }

// colorwash 10
	colorwash.szx = outputsz.width;
	colorwash.szy = outputsz.height;
	if (mode & 0x10) { colorwash(L, _scratch_floats[0]->space(), currsz, currlinesz, _scratch_floats[1]->space(), currsz, currlinesz); swap_scratch(_scratch_floats); }

// adjust_contrast 20
	adjust_contrast.szx = outputsz.width;
	adjust_contrast.szy = outputsz.height;
	if (mode & 0x20) { adjust_contrast(L, _scratch_floats[0]->space(), currsz, currlinesz, _scratch_floats[1]->space(), currsz, currlinesz); swap_scratch(_scratch_floats); }

	noop(L, _scratch_floats[0]->space(), currsz, currlinesz, output, currsz, currlinesz); swap_scratch(_scratch_floats);
// Now, verify that currsz and outputsz are the same and that outputlinesz and currlinesz are the same.
	luax21_assert(L, (outputsz.width == currsz.width), "The outputsz.width did not match the currsz.width! Odds are there's a memory problem imminent.");
	luax21_assert(L, (outputsz.height == currsz.height), "The outputsz.height did not match the currsz.height! Odds are there's a memory problem imminent.");
	luax21_assert(L, (outputlinesz == currlinesz), "The outputlinesz did not match the currlinesz! Odds are there's a memory problem imminent.");
}

void pipeline_impl::operator()(lua_State *L, int random, const THCudaTensor *inputbatch, THCudaTensor *outputbatch) {
	THCState *thcstate = cutorch_getstate(L);
	luax21_assert(L, (THCudaTensor_isContiguous(thcstate, inputbatch)), "pipeline expects a contiguous input minibatch.");
	luax21_assert(L, (THCudaTensor_isContiguous(thcstate, outputbatch)), "pipeline expects a contiguous output minibatch.");
	luax21_assert(L, (inputbatch->nDimension == 4), "pipeline expects an input minibatch of RGB images. (%d != 4)", inputbatch->nDimension);
	luax21_assert(L, (outputbatch->nDimension == 4), "pipeline expects an output minibatch of RGB images. (%d != 4)", outputbatch->nDimension);
	luax21_assert(L, (inputbatch->size[0] == outputbatch->size[0]), "pipeline expects the input and output to have the same minibatch size");
	luax21_assert(L, (inputbatch->size[1] == inputbatch->size[2]), "pipeline expects input minibatches to be square.");
	luax21_assert(L, (outputbatch->size[1] == outputbatch->size[2]), "pipeline expects output minibatches to be square.");
	luax21_assert(L, (inputbatch->size[3] == 3), "pipeline expects input minibatch of RGB images in form [Y, X, C]");
	luax21_assert(L, (outputbatch->size[3] == 3), "pipeline expects output minibatch of RGB images in form [Y, X, C]");

	int m = inputbatch->size[0];
	NppiSize inputsz;
	NppiSize outputsz;

	inputsz.width = inputbatch->size[2];
	inputsz.height = inputbatch->size[1];

	outputsz.width = outputbatch->size[2];
	outputsz.height = outputbatch->size[1];

	for (int i = 0; i < m; ++i) {
		const float *input = inputbatch->storage->data + i * inputbatch->stride[0];
		float *output = outputbatch->storage->data + i * outputbatch->stride[0];
		int  inputlinesz =  inputbatch->stride[1] * sizeof(( *inputbatch->storage->data));
		int outputlinesz = outputbatch->stride[1] * sizeof((*outputbatch->storage->data));
		if (random) random_filter(L, input, inputsz, inputlinesz, output, outputsz, outputlinesz);
		else center_filter(L, input, inputsz, inputlinesz, output, outputsz, outputlinesz);
	}
}

void noop_filter::operator()(lua_State *L, const Npp32f *input, NppiSize inputsz, int inputlinesz, Npp32f *output, NppiSize& outputsz, int& outputlinesz) {
	outputsz.width  = inputsz.width;
	outputsz.height = inputsz.height;
	outputlinesz = outputsz.width * sizeof((*output)) * 3;

	const Npp32f *src = input;
	int ss = nppiCopy_32f_C3R(src, inputlinesz, output, outputlinesz, outputsz);
	luax21_assert(L, ss >= 0, "nppiCopy_32f_C3R returned %d", ss);
}

void center_crop_filter::operator()(lua_State *L, const Npp32f *input, NppiSize inputsz, int inputlinesz, Npp32f *output, NppiSize& outputsz, int& outputlinesz) {
	luax21_assert(L, szx <= inputsz.width, "center_crop called with szx (%d) > input width (%d)!", szx, inputsz.width);
	luax21_assert(L, szy <= inputsz.height, "center_crop called with szy (%d) > input height (%d)!", szy, inputsz.height);
	luax21_assert(L, szx >= 0, "center_crop was called with szx=%d", szx);
	luax21_assert(L, szy >= 0, "center_crop was called with szy=%d", szy);

	outputsz.width  = szx;
	outputsz.height = szy;
	outputlinesz = outputsz.width * sizeof((*output)) * 3;

	int starty = (inputsz.height - outputsz.height) / 2.;
	int startx = (inputsz.width - outputsz.width) / 2.;

	const Npp32f *src = input + starty * inputlinesz/sizeof((*input)) + startx * 3;
	int ss = nppiCopy_32f_C3R(src, inputlinesz, output, outputlinesz, outputsz);
	luax21_assert(L, ss >= 0, "nppiCopy_32f_C3R returned %d", ss);
}

void random_square_crop_filter::operator()(lua_State *L, const Npp32f *input, NppiSize inputsz, int inputlinesz, Npp32f *output, NppiSize& outputsz, int& outputlinesz) {
	luax21_assert(L, minszx <= inputsz.width, "random_square_crop called with minszx (%d) > input width (%d)!", minszx, inputsz.width);
	luax21_assert(L, minszy <= inputsz.height, "random_square_crop called with minszy (%d) > input height (%d)!", minszy, inputsz.height);
	luax21_assert(L, minszx >= 0, "random_square_crop was called with minszx=%d", minszx);
	luax21_assert(L, minszy >= 0, "random_square_crop was called with minszy=%d", minszy);
// min this can be is minsz. Check.
// max this can be is minsz + (inputsz.width - minsz - 1)
// = minsz + inputsz.width - minsz - 1
// = inputsz.width - 1
	outputsz.width  = minszx + lrand48() % (inputsz.width - minszx);
	outputsz.height = outputsz.width;
	outputlinesz = outputsz.width * sizeof((*output)) * 3;

// Now, given that we've chosen a particular width, we have a range
// 0..(inputsz.width-outputsz.width) that we can choose from.
	int starty = lrand48() % (inputsz.height - outputsz.height);
	int startx = lrand48() % (inputsz.width - outputsz.width);

	const Npp32f *src = input + starty * inputlinesz/sizeof((*input)) + startx * 3;
	int ss = nppiCopy_32f_C3R(src, inputlinesz, output, outputlinesz, outputsz);
	luax21_assert(L, ss >= 0, "nppiCopy_32f_C3R returned %d", ss);
}

void gaussian_blur_filter::operator()(lua_State *L, const Npp32f *input, NppiSize inputsz, int inputlinesz, Npp32f *output, NppiSize& outputsz, int& outputlinesz) {
	int halfkw = 1 + lrand48() % 5; // choose a halfkw between 1 and 5 inclusive.
	int kw = (1 + 2 * halfkw);
	int kernelsz = kw * kw;
	if ((_kernel == NULL) || (_kernel->size() < kernelsz)) {
		if (_kernel) delete _kernel; _kernel = NULL;
		_kernel = new scratch<Npp32f>(kernelsz);
	}

/*
 * our kernel will be something like
 * {
 * 	{ b,b,b,b,b },
 * 	{ b,b,b,b,b },
 * 	{ b,b,a,b,b },
 * 	{ b,b,b,b,b },
 * 	{ b,b,b,b,b },
 * }
 * in our case halfkw = 2
 * kw = 5
 * kernelsz = 25
 * a + b * (kernelsz-1) = 1.0
 * b * (kernelsz - 1) = 1.0 - a
 * b = (1.0 - a) / (kernelsz - 1)
 */

	Npp32f a = lerp(drand48(),0,1.5);
	Npp32f b = (1.0-a) / (kernelsz - 1.0);

	Npp32f *cpukernel = new Npp32f[kernelsz];
	for (int i = 0; i < kernelsz; ++i) cpukernel[i] = b;
	cpukernel[halfkw * (kw + 1)] = a;
	luax21_assert(L, 0 == cudaMemcpy(_kernel->space(), cpukernel, sizeof((*cpukernel)) * kernelsz, cudaMemcpyHostToDevice), "cudaMemcpy failed in gaussian_blur_filter!");
	delete[] cpukernel; cpukernel = NULL;

	outputsz.width  = inputsz.width - 2 * halfkw;
	outputsz.height = inputsz.height - 2 * halfkw;
	outputlinesz = outputsz.width * sizeof((*output)) * 3;

	int starty = halfkw;
	int startx = halfkw;

// NppStatus nppiFilter_32f_C3R (const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32f *pKernel, NppiSize oKernelSize, NppiPoint oAnchor)
	NppiSize kernelsize = { kw, kw };
	NppiPoint kernelanchor = { halfkw, halfkw };

	const Npp32f *src = input + starty * inputlinesz/sizeof((*input)) + startx * 3;
	int ss = nppiFilter_32f_C3R(src, inputlinesz, output, outputlinesz, outputsz, _kernel->space(), kernelsize, kernelanchor);
	luax21_assert(L, ss >= 0, "nppiFilter_32f_C3R returned %d", ss);
}

void rotatecrop_filter::operator()(lua_State *L, const Npp32f *input, NppiSize inputsz, int inputlinesz, Npp32f *output, NppiSize& outputsz, int& outputlinesz) {
	luax21_assert(L, inputsz.width == inputsz.height, "rotatecrop demands an square image (%d != %d)!", inputsz.width, inputsz.height);

	// So, if we've rotated an image of size sz _squared_ then
	// our output should be of size sz / (Sin[t]+Cos[t])
	// But, we don't care if it's above 90 degrees, (in which case, it's
	// just a corner shift, etc. etc. etc.) and at the end, we come up with
	// sz / (Sqrt[2] Cos[t - Pi/4 - Floor[(2 t / Pi)] Pi / 2])

	Npp32f theta = lerp(drand48(),-M_PI/8.,M_PI/8.);
	int sz = inputsz.width;
	double denominator = sqrt(2.0) * cos(theta - M_PI / 4.0 - floor(2 * theta / M_PI) * M_PI / 2.0);
	int rotatedsz = floor(sz / denominator);

	outputsz.width  = rotatedsz;
	outputsz.height = rotatedsz;
	outputlinesz = outputsz.width * sizeof((*output)) * 3;

	NppiRect imgroi = { 0, 0, sz, sz };
	NppiRect outputroi = { 0, 0, outputsz.width, outputsz.height };
// This was laboriously calculated.
	double shiftx =  (sz - sz * cos(theta) - sz * sin(theta)) / 2. - (sz - rotatedsz)/2.;
	double shifty =  (sz + sz * sin(theta) - sz * cos(theta)) / 2. - (sz - rotatedsz)/2.;

	int ss = nppiRotate_32f_C3R(input, inputsz, inputlinesz, imgroi, output, outputlinesz, outputroi, theta * 180.0 / M_PI, shiftx, shifty, interpolationmode);
	luax21_assert(L, ss >= 0, "nppiRotate_32f_C3R returned %d", ss);
}

void scale_filter::operator()(lua_State *L, const Npp32f *input, NppiSize inputsz, int inputlinesz, Npp32f *output, NppiSize& outputsz, int& outputlinesz) {
	luax21_assert(L, scaledx >= 0, "scale was called with scaledx=%d", scaledx);
	luax21_assert(L, scaledy >= 0, "scale was called with scaledy=%d", scaledy);

// So, we need to scale our input to squaresz.
	outputsz.width = scaledx;
	outputsz.height = scaledy;
	outputlinesz = outputsz.width * sizeof((*output)) * 3;

	NppiRect inputroi = { 0,0, inputsz.width, inputsz.height };
	NppiRect outputroi = { 0,0, outputsz.width, outputsz.height };

	double scalefactor = 1.0;
	double shiftx = 0.0;
	double shifty = 0.0;
	if (inputsz.height * outputsz.width > inputsz.width * outputsz.height) { // aspect ratio of the input > aspect ratio of output
// scale height
		scalefactor = outputsz.height * 1.0 / inputsz.height;
		shiftx = ((outputsz.width * inputsz.height * 1.0 / outputsz.height) - inputsz.width) / 2.0;
	}
	else {
// scale width
		scalefactor = outputsz.width * 1.0 / inputsz.width;
		shifty = ((outputsz.height * inputsz.width * 1.0 / outputsz.width) - inputsz.height) / 2.0;
	}

	int ss = nppiResizeSqrPixel_32f_C3R(input, inputsz, inputlinesz, inputroi, output, outputlinesz, outputroi, scalefactor, scalefactor, shiftx, shifty, interpolationmode);
	luax21_assert(L, ss >= 0, "nppiResizeSqrPixel_32f_C3R returned %d", ss);
}

void horizontalflip_filter::operator()(lua_State *L, const Npp32f *input, NppiSize inputsz, int inputlinesz, Npp32f *output, NppiSize& outputsz, int& outputlinesz) {
	outputsz.width = szx;
	outputsz.height = szy;
	outputlinesz = outputsz.width * sizeof((*output)) * 3;

	if (drand48() > 0.5) {
		int ss = nppiMirror_32f_C3R(input, inputlinesz, output, outputlinesz, inputsz, NPP_VERTICAL_AXIS);
		luax21_assert(L, ss >= 0, "nppiMirror_32f_C3R returned %d", ss);
	}
	else { // Just copy the input.
		int ss = nppiCopy_32f_C3R(input, inputlinesz, output, outputlinesz, inputsz);
		luax21_assert(L, ss >= 0, "nppiCopy_32f_C3R returned %d", ss);
	}
}

void colorwash_filter::operator()(lua_State *L, const Npp32f *input, NppiSize inputsz, int inputlinesz, Npp32f *output, NppiSize& outputsz, int& outputlinesz) {
	outputsz.width = szx;
	outputsz.height = szy;
	outputlinesz = outputsz.width * sizeof((*output)) * 3;

// Make p between 0.8 and 1.2
	double p = lerp(drand48(),0.8,1.2);
	if (p < 0.0) p = 0.0;
	if (p > 1.0) p = 1.0;

	float twist[3][4];
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 4; ++j) {
			twist[i][j] = 0.0;
			if (i == j) twist[i][j] = (2.0 * p + 1.0) / 3.;
			else twist[i][j] = (1.0 - p) / 3.;
		}
	}

	int ss = nppiColorTwist_32f_C3R(
		input,
		inputlinesz,
		output,
		outputlinesz,
		outputsz,
		twist
	);
	luax21_assert(L, ss >= 0, "nppiColorTwist_32f_C3R returned %d", ss);
}

void adjust_contrast_filter::operator()(lua_State *L, const Npp32f *input, NppiSize inputsz, int inputlinesz, Npp32f *output, NppiSize& outputsz, int& outputlinesz) {
	outputsz.width = szx;
	outputsz.height = szy;
	outputlinesz = outputsz.width * sizeof((*output)) * 3;

// We need a random contrast and a random brightness.
	double contrast = lerp(drand48(),0.5,1.5);
	double brightness = lerp(drand48(),-1.,1.);

// So, first we have to find the mean and standard deviations of the pixels.
	int buffersz = -1;
	int ss = nppiMeanStdDevGetBufferHostSize_32f_C3CR(inputsz, &buffersz);
	luax21_assert(L, ss == 0, "nppiMeanStdDevGetBufferHostSize_32f_C3CR returned %d", ss);

	if ((_scratchspc == NULL) || (_scratchspc->size() < buffersz)) {
		if (_scratchspc) delete _scratchspc; _scratchspc = NULL;
		_scratchspc = new scratch<Npp8u>(buffersz);
	}

	double *gpumean = _mean->space();
	double *gpustd = _std->space();

	for (int i = 0; i < 3; ++i) {
		ss = nppiMean_StdDev_32f_C3CR(
			input,
			inputlinesz,
			inputsz,
			(1+i),
			_scratchspc->space(),
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
			if (j == 4) twist[i][j] = cpumean[i] - contrast * cpumean[i] + brightness * cpustd[i];
			else if (i == j) twist[i][j] = contrast;
			else twist[i][j] = 0;
		}
	}

	ss = nppiColorTwist_32f_C3R(
		input,
		inputlinesz,
		output,
		outputlinesz,
		outputsz,
		twist
	);
	luax21_assert(L, ss >= 0, "nppiColorTwist_32f_C3R returned %d", ss);
}
