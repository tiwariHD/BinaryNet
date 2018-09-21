from __future__ import absolute_import, print_function, division
import os
import logging
from six import integer_types
from six.moves import StringIO, reduce
import theano
from theano import Apply
from theano import tensor
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous)
from theano.tensor import as_tensor_variable
_logger = logging.getLogger(__name__)


class BaseGpuCorrMM(GpuOp):
    """
    Base class for `GpuCorrMM`, `GpuCorrMM_gradWeights` and
    `GpuCorrMM_gradInputs`. Cannot be used directly.

    Parameters
    ----------
    border_mode : {'valid', 'full', 'half'}
        Additionally, the padding size could be directly specified by an integer
        or a pair of integers
    subsample
        Perform subsampling of the output (default: (1, 1)).
    filter_dilation
        Perform subsampling of the input, also known as dilation (default: (1, 1)).
    pad
        *deprecated*, now you should always use border_mode.
    """

    check_broadcast = False
    __props__ = ('border_mode', 'subsample', 'filter_dilation')

    def __init__(self, border_mode="valid", subsample=(1, 1),
                 filter_dilation=(1, 1), pad=None, binary=False):
        if pad is not None:
            _logger.warning(
                'do not use pad for BaseGpuCorrMM; please set padding in '
                'border_mode parameter, see the docstring for more details')
            if border_mode != "valid":
                raise ValueError("border_mode must be 'valid' if pad is given")
            border_mode = pad
        if isinstance(border_mode, integer_types):
            border_mode = (border_mode, border_mode)
        if isinstance(border_mode, tuple):
            pad_h, pad_w = map(int, border_mode)
            border_mode = (pad_h, pad_w)
        if not ((isinstance(border_mode, tuple) and min(border_mode) >= 0) or
                border_mode in ('valid', 'full', 'half')):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a pair of'
                ' integers'.format(border_mode))
        self.border_mode = border_mode
        if len(subsample) != 2:
            raise ValueError("subsample must have two elements")
        if len(filter_dilation) != 2:
            raise ValueError("filter_dilation must have two elements")
        self.subsample = tuple(subsample)
        self.filter_dilation = tuple(filter_dilation)
        self.binary = binary

    @property
    def pad(self):
        if self.border_mode != 'valid':
            return self.border_mode
        return (0, 0)

    def __str__(self):
        return '%s{%s, %s, %s}' % (
            self.__class__.__name__,
            self.border_mode,
            str(self.subsample),
            str(self.filter_dilation))

    def flops(self, inp, outp):
        """
        Useful with the hack in profiling to print the MFlops.

        """
        # if the output shape is correct, then this gives the correct
        # flops for any direction, sampling, padding, and border mode
        inputs, filters = inp
        outputs, = outp
        assert inputs[1] == filters[1]
        # nb mul and add by output pixel
        flops = filters[2] * filters[3] * 2
        # nb flops by output image
        flops *= outputs[2] * outputs[3]
        # nb patch multiplied
        flops *= inputs[1] * filters[0] * inputs[0]
        return flops

    def c_headers(self):
        return ['cuda_ndarray.cuh', '<stdio.h>']

    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (1, 30)

    def c_support_code_apply(self, node, nodename):
        # REMEMBER TO RAISE c_code_cache_version when changing any of
        # these files
        files = ['corr_gemm.cu']
        codes = [open(os.path.join(os.path.split(__file__)[0], f)).read()
                 for f in files]
        return reduce(str.__add__, codes)

    def c_code_helper(self, bottom, weights, top, direction, sub, height=None, width=None):
        """
        This generates the C code for GpuCorrMM (direction="forward"),
        GpuCorrMM_gradWeights (direction="backprop weights"), and
        GpuCorrMM_gradInputs (direction="backprop inputs").
        Depending on the direction, one of bottom, weights, top will
        receive the output, while the other two serve as inputs.

        Parameters
        ----------
        bottom
            Variable name of the input images in the forward pass,
            or the gradient of the input images in backprop wrt. inputs
        weights
            Variable name of the filters in the forward pass,
            or the gradient of the filters in backprop wrt. weights
        top
            Variable name of the output images / feature maps in the
            forward pass, or the gradient of the outputs in the backprop passes
        direction : {'forward', 'backprop weights', 'backprop inputs'}
            "forward" to correlate bottom with weights and store results in top,
            "backprop weights" to do a valid convolution of bottom with top
            (swapping the first two dimensions) and store results in weights,
            and "backprop inputs" to do a full convolution of top with weights
            (swapping the first two dimensions) and store results in bottom.
        sub
            Dictionary of substitutions useable to help generating the C code.
        height
            Required if self.subsample[0] != 1, a variable giving the height of
            the filters for direction="backprop weights" or the height of the
            input images for direction="backprop inputs".
            Required if self.border_mode == 'half', a variable giving the height
            of the filters for direction="backprop weights".
            Not required otherwise, but if a value is given this will be checked.
        width
            Required if self.subsample[1] != 1, a variable giving the width of
            the filters for direction="backprop weights" or the width of the
            input images for direction="backprop inputs".
            Required if self.border_mode == 'half', a variable giving the width
            of the filters for direction="backprop weights".
            Not required otherwise, but if a value is given this will be checked.

        """
        callBinary = 0
        if self.binary == True:
            callBinary = 1

        print("callbinary = " + str(callBinary))

        dH, dW = self.subsample
        dilH, dilW = self.filter_dilation
        if self.border_mode == "half":
            padH = padW = -1
        elif self.border_mode == "full":
            padH = padW = -2
        elif isinstance(self.border_mode, tuple):
            padH, padW = self.border_mode
        else:
            assert self.border_mode == "valid"
            padH = padW = 0
        if direction == "forward":
            direction = 0
            out = top
        elif direction == "backprop weights":
            direction = 1
            out = weights
        elif direction == "backprop inputs":
            direction = 2
            out = bottom
        else:
            raise ValueError("direction must be one of 'forward', "
                             "'backprop weights', 'backprop inputs'")
        # When subsampling, we cannot unambiguously infer the height and width
        # of bottom and weights from top, so we require them to be given.
        # Similarly, when pad="half", we cannot infer the weight size.
        if height:
            height = '(*(npy_int*)(PyArray_DATA(%s)))' % height
        else:
            if ((direction != 0) and (dH != 1)) or ((direction == 1) and (padH == -1)):
                raise ValueError("height must be given for backprop with vertical sampling or pad='half'")
            height = '-1'
        if width:
            width = '(*(npy_int*)(PyArray_DATA(%s)))' % width
        else:
            if ((direction != 0) and (dW != 1)) or ((direction == 1) and (padW == -1)):
                raise ValueError("width must be given for backprop with horizontal sampling or pad='half'")
            width = '-1'
        sub = sub.copy()
        sub.update(locals())

        return """
    // Mandatory args
    int direction = %(direction)s;  // forward, bprop weights, bprop inputs

    // Optional args
    int dH = %(dH)s;
    int dW = %(dW)s;
    int dilH = %(dilH)s;
    int dilW = %(dilW)s;
    int padH = %(padH)s;
    int padW = %(padW)s;
    int callBinary = %(callBinary)s;

    CudaNdarray * bottom = %(bottom)s;
    CudaNdarray * weights = %(weights)s;
    CudaNdarray * top = %(top)s;
    CudaNdarray * out2 = NULL;

    // Obtain or infer kernel width and height
    // (we need to know it early to be able to handle auto-padding)
    int kH, kW, dil_kH, dil_kW;
    if (direction != 1) {
        // weight is an input variable, we can just read its shape
        kH = CudaNdarray_HOST_DIMS(weights)[2];
        kW = CudaNdarray_HOST_DIMS(weights)[3];
    }
    else {
        if (%(height)s != -1) {
            // kernel height is specified (perhaps vertical subsampling or half padding)
            kH = %(height)s;
        }
        else if (padH == -2) {
            // vertical full padding, we can infer the kernel height
            kH = (2 - CudaNdarray_HOST_DIMS(bottom)[2] + (CudaNdarray_HOST_DIMS(top)[2] - 1)*dH - 1) / dilH + 1;
        }
        else {
            // explicit padding, we can infer the kernel height
            kH = (CudaNdarray_HOST_DIMS(bottom)[2] + 2*padH - (CudaNdarray_HOST_DIMS(top)[2] - 1)*dH - 1) / dilH + 1 ;
        }
        if (%(width)s != -1) {
            kW = %(width)s;
        }
        else if (padW == -2) {
            kW = (2 - CudaNdarray_HOST_DIMS(bottom)[3] + (CudaNdarray_HOST_DIMS(top)[3] - 1) * dW - 1) / dilW + 1;
        }
        else {
            kW = (CudaNdarray_HOST_DIMS(bottom)[3] + 2*padW - (CudaNdarray_HOST_DIMS(top)[3] - 1) * dW - 1) / dilW + 1;
        }
    }

    // Implicit dilated kernel size
    dil_kH = (kH - 1) * dilH + 1;
    dil_kW = (kW - 1) * dilW + 1;

    // Auto-padding if requested
    if (padH == -1) {  // vertical half padding
        padH = dil_kH / 2;
    }
    else if (padH == -2) {  // vertical full padding
        padH = dil_kH - 1;
    }
    else if (padH < 0) {
        PyErr_SetString(PyExc_ValueError, "BaseGpuCorrMM: padH must be >= -2");
        %(fail)s
    }
    if (padW == -1) {  // horizontal half padding
        padW = dil_kW / 2;
    }
    else if (padW == -2) {  // horizontal full padding
        padW = dil_kW - 1;
    }
    else if (padW < 0) {
        PyErr_SetString(PyExc_ValueError, "BaseGpuCorrMM: padW must be >= -2");
        %(fail)s
    }

    // Infer output shape
    int out_dim[4];
    switch(direction) {
    case 0:  // forward pass
        // output is top: (batchsize, num_filters, height, width)
        // height and width: top = (bottom + 2*pad - ((weight-1)*dil + 1)) / sample + 1
        out_dim[0] = CudaNdarray_HOST_DIMS(bottom)[0];
        out_dim[1] = CudaNdarray_HOST_DIMS(weights)[0];
        out_dim[2] = (CudaNdarray_HOST_DIMS(bottom)[2] + 2*padH - ((CudaNdarray_HOST_DIMS(weights)[2]-1)*dilH + 1)) / dH + 1;
        out_dim[3] = (CudaNdarray_HOST_DIMS(bottom)[3] + 2*padW - ((CudaNdarray_HOST_DIMS(weights)[3]-1)*dilW + 1)) / dW + 1;
        if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0)
        {
            PyErr_Format(PyExc_ValueError,
                         "GpuCorrMM: impossible output shape\\n"
                         "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                         "  weights shape: %%ld x %%ld x %%ld x %%ld\\n"
                         "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                         CudaNdarray_HOST_DIMS(bottom)[0], CudaNdarray_HOST_DIMS(bottom)[1],
                         CudaNdarray_HOST_DIMS(bottom)[2], CudaNdarray_HOST_DIMS(bottom)[3],
                         CudaNdarray_HOST_DIMS(weights)[0], CudaNdarray_HOST_DIMS(weights)[1],
                         CudaNdarray_HOST_DIMS(weights)[2], CudaNdarray_HOST_DIMS(weights)[3],
                         out_dim[0], out_dim[1], out_dim[2], out_dim[3]);
            %(fail)s
        }
        break;
    case 1:  // backprop wrt. weights
        // output is weights: (num_filters, num_channels, height, width)
        // height and width: weights = (bottom + 2*pad - (top - 1) * sample - 1) / dil + 1
        out_dim[0] = CudaNdarray_HOST_DIMS(top)[1];
        out_dim[1] = CudaNdarray_HOST_DIMS(bottom)[1];
        out_dim[2] = kH;  // already inferred further above
        out_dim[3] = kW;  // how convenient
        if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0)
        {
            PyErr_Format(PyExc_ValueError,
                         "GpuCorrMM backprop wrt. weights: impossible output shape\\n"
                         "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                         "  weights shape: %%ld x %%ld x %%ld x %%ld\\n"
                         "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                         CudaNdarray_HOST_DIMS(bottom)[0], CudaNdarray_HOST_DIMS(bottom)[1],
                         CudaNdarray_HOST_DIMS(bottom)[2], CudaNdarray_HOST_DIMS(bottom)[3],
                         out_dim[0], out_dim[1], out_dim[2], out_dim[3],
                         CudaNdarray_HOST_DIMS(top)[0], CudaNdarray_HOST_DIMS(top)[1],
                         CudaNdarray_HOST_DIMS(top)[2], CudaNdarray_HOST_DIMS(top)[3]);
            %(fail)s
        }
        break;
    case 2:  // backprop wrt. inputs
        // output is bottom: (batchsize, num_channels, height, width)
        // height and width: bottom = (top - 1) * sample + (weights-1)*dil + 1 - 2*pad
        out_dim[0] = CudaNdarray_HOST_DIMS(top)[0];
        out_dim[1] = CudaNdarray_HOST_DIMS(weights)[1];
        out_dim[2] = (%(height)s != -1) ? %(height)s : (CudaNdarray_HOST_DIMS(top)[2] - 1) * dH + (CudaNdarray_HOST_DIMS(weights)[2]-1)*dilH + 1 - 2*padH;
        out_dim[3] = (%(width)s != -1) ? %(width)s : (CudaNdarray_HOST_DIMS(top)[3] - 1) * dW + (CudaNdarray_HOST_DIMS(weights)[3]-1)*dilW + 1 - 2*padW;
        if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0)
        {
            PyErr_Format(PyExc_ValueError,
                         "GpuCorrMM backprop wrt. inputs: impossible output shape\\n"
                         "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                         "  weight shape: %%ld x %%ld x %%ld x %%ld\\n"
                         "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                         out_dim[0], out_dim[1], out_dim[2], out_dim[3],
                         CudaNdarray_HOST_DIMS(weights)[0], CudaNdarray_HOST_DIMS(weights)[1],
                         CudaNdarray_HOST_DIMS(weights)[2], CudaNdarray_HOST_DIMS(weights)[3],
                         CudaNdarray_HOST_DIMS(top)[0], CudaNdarray_HOST_DIMS(top)[1],
                         CudaNdarray_HOST_DIMS(top)[2], CudaNdarray_HOST_DIMS(top)[3]);
            %(fail)s
        }
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "BaseGpuCorrMM: direction must be 0, 1, or 2\\n");
        %(fail)s
    }

    // Prepare output array
    if ( !(%(out)s
           && %(out)s->nd==4
           && CudaNdarray_is_c_contiguous(%(out)s)
           && CudaNdarray_HOST_DIMS(%(out)s)[0]==out_dim[0]
           && CudaNdarray_HOST_DIMS(%(out)s)[1]==out_dim[1]
           && CudaNdarray_HOST_DIMS(%(out)s)[2]==out_dim[2]
           && CudaNdarray_HOST_DIMS(%(out)s)[3]==out_dim[3]))
    {
        Py_XDECREF(%(out)s);
        %(out)s = (CudaNdarray*)CudaNdarray_NewDims(4,out_dim);
        if (NULL == %(out)s)
        {
            PyErr_Format(PyExc_RuntimeError,
                    "BaseGpuCorrMM: Failed to allocate output of %%d x %%d x %%d x %%d",
                    out_dim[0], out_dim[1], out_dim[2], out_dim[3]);
            %(fail)s
        }
    }

    // Call CUDA code
    out2 = corrMMWrapper(%(bottom)s, %(weights)s, %(top)s, direction, dH, dW, dilH, dilW, padH, padW, callBinary);
    if (out2==NULL)
    {
        %(fail)s
    }
    assert (out2 == %(out)s);

""" % sub


class GpuCorrMM(BaseGpuCorrMM):
    """
    GPU correlation implementation using Matrix Multiplication.

    Parameters
    ----------
    border_mode
        The width of a border of implicit zeros to pad the
        input with. Must be a tuple with 2 elements giving the numbers of rows
        and columns to pad on each side, or a single integer to pad the same
        on all sides, or a string shortcut setting the padding at runtime:
        ``'valid'`` for ``(0, 0)`` (valid convolution, no padding), ``'full'``
        for ``(kernel_rows - 1, kernel_columns - 1)`` (full convolution),
        ``'half'`` for ``(kernel_rows // 2, kernel_columns // 2)`` (same
        convolution for odd-sized kernels). Note that the two widths are each
        applied twice, once per side (left and right, top and bottom).
    subsample
        The subsample operation applied to each output image.
        Should be a tuple with 2 elements.
        `(sv, sh)` is equivalent to `GpuCorrMM(...)(...)[:,:,::sv, ::sh]`,
        but faster.
        Set to `(1, 1)` to disable subsampling.
    filter_dilation
        The filter dilation operation applied to each input image.
        Should be a tuple with 2 elements.
        Set to `(1, 1)` to disable filter dilation.
    pad
        Deprecated alias for `border_mode`.

    Notes
    -----
    Currently, the Op requires the inputs, filters and outputs to be
    C-contiguous. Use :func:`gpu_contiguous
    <theano.sandbox.cuda.basic_ops.gpu_contiguous>` on these arguments
    if needed.

    You can either enable the Theano flag `optimizer_including=conv_gemm`
    to automatically replace all convolution operations with `GpuCorrMM`
    or one of its gradients, or you can use it as a replacement for
    :func:`conv2d <theano.tensor.nnet.conv.conv2d>`, called as
    `GpuCorrMM(subsample=...)(image, filters)`. The latter is currently
    faster, but note that it computes a correlation -- if you need to
    compute a convolution, flip the filters as `filters[:,:,::-1,::-1]`.

    ..warning:: For 700 series Nvidia GPUs of compute capability 3.5 and CUDA 5.0
        to 6.0, there is a bug in CUBLAS' matrix multiplication function that
        can make GpuCorrMM or its gradients crash for some input and filter
        shapes. So if you have a Tesla K20, Tesla K40, Quadro K6000, GeForce GT
        640 (DDR5), GeForce GTX 780 (or Ti), GeForce GTX TITAN (or Black or Z)
        and experience a crash, switching to CUDA 6.5 or CUDA 4.2 should fix it.
        If this is not possible, changing the input or filter shapes (e.g., the
        batchsize or number of filters) may also work around the CUBLAS bug.

    """
    def __init__(self, border_mode="valid",
                 subsample=(1, 1),
                 filter_dilation=(1, 1),
                 pad=None,
                 binary=False):
        super(GpuCorrMM, self).__init__(border_mode, subsample,
                                        filter_dilation, pad, binary)

    def make_node(self, img, kern):
        img = as_cuda_ndarray_variable(img)
        kern = as_cuda_ndarray_variable(kern)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')

        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False]
        return Apply(self, [img, kern], [CudaNdarrayType(broadcastable)()])

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, weights = inp
        top, = out_
        direction = "forward"
        return super(GpuCorrMM, self).c_code_helper(bottom, weights, top, direction, sub)

    def grad(self, inp, grads):
        bottom, weights = inp
        top, = grads
        top = gpu_contiguous(top)
        d_bottom = GpuCorrMM_gradInputs(self.border_mode,
                                        self.subsample,
                                        self.filter_dilation)(
            weights, top, bottom.shape[-2:])
        d_weights = GpuCorrMM_gradWeights(self.border_mode,
                                          self.subsample,
                                          self.filter_dilation)(
            bottom, top, weights.shape[-2:])
        return d_bottom, d_weights


class GpuCorrMM_gradWeights(BaseGpuCorrMM):
    """
    Gradient wrt. filters for `GpuCorrMM`.

    Notes
    -----
    You will not want to use this directly, but rely on Theano's automatic
    differentiation or graph optimization to use it as needed.

    """

    def __init__(self, border_mode="valid",
                 subsample=(1, 1),
                 filter_dilation=(1, 1),
                 pad=None):
        super(GpuCorrMM_gradWeights, self).__init__(border_mode,
                                                    subsample,
                                                    filter_dilation,
                                                    pad)

    def make_node(self, img, topgrad, shape=None):
        img = as_cuda_ndarray_variable(img)
        topgrad = as_cuda_ndarray_variable(topgrad)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')
        if shape is None:
            if self.subsample != (1, 1) or self.border_mode == "half":
                raise ValueError('shape must be given if subsample != (1, 1)'
                                 ' or border_mode == "half"')
            height_width = []
        else:
            height_width = [shape[0], shape[1]]
            assert shape[0].ndim == 0
            assert shape[1].ndim == 0

        broadcastable = [topgrad.type.broadcastable[1], img.type.broadcastable[1],
                         False, False]
        return Apply(self, [img, topgrad] + height_width, [CudaNdarrayType(broadcastable)()])

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, top = inp[:2]
        height, width = inp[2:] or (None, None)
        weights, = out_
        direction = "backprop weights"
        return super(GpuCorrMM_gradWeights, self).c_code_helper(bottom, weights, top, direction, sub, height, width)

    def grad(self, inp, grads):
        bottom, top = inp[:2]
        weights, = grads
        weights = gpu_contiguous(weights)
        d_bottom = GpuCorrMM_gradInputs(self.border_mode,
                                        self.subsample,
                                        self.filter_dilation)(weights,
                                                              top,
                                                              bottom.shape[-2:])
        d_top = GpuCorrMM(
            self.border_mode, self.subsample, self.filter_dilation)(bottom, weights)
        d_height_width = (
            theano.gradient.DisconnectedType()(),
            ) * 2 if len(inp) == 4 else ()
        return (d_bottom, d_top) + d_height_width

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0]]  # no connection to height, width


class GpuCorrMM_gradInputs(BaseGpuCorrMM):
    """
    Gradient wrt. inputs for `GpuCorrMM`.

    Notes
    -----
    You will not want to use this directly, but rely on Theano's automatic
    differentiation or graph optimization to use it as needed.

    """

    def __init__(self, border_mode="valid",
                 subsample=(1, 1),
                 filter_dilation=(1, 1),
                 pad=None):
        super(GpuCorrMM_gradInputs, self).__init__(border_mode, subsample,
                                                   filter_dilation, pad)

    def make_node(self, kern, topgrad, shape=None):
        kern = as_cuda_ndarray_variable(kern)
        topgrad = as_cuda_ndarray_variable(topgrad)
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')
        if shape is None:
            if self.subsample != (1, 1):
                raise ValueError('shape must be given if subsample != (1, 1)')
            height_width = []
        else:
            height_width = [shape[0], shape[1]]
            assert shape[0].ndim == 0
            assert shape[1].ndim == 0

        broadcastable = [topgrad.type.broadcastable[0], kern.type.broadcastable[1],
                         False, False]
        return Apply(self, [kern, topgrad] + height_width, [CudaNdarrayType(broadcastable)()])

    def c_code(self, node, nodename, inp, out_, sub):
        weights, top = inp[:2]
        height, width = inp[2:] or (None, None)
        bottom, = out_
        direction = "backprop inputs"
        return super(GpuCorrMM_gradInputs, self).c_code_helper(bottom, weights, top, direction, sub, height, width)

    def grad(self, inp, grads):
        weights, top = inp[:2]
        bottom, = grads
        bottom = gpu_contiguous(bottom)
        d_weights = GpuCorrMM_gradWeights(self.border_mode,
                                          self.subsample,
                                          self.filter_dilation)(bottom,
                                                                top,
                                                                weights.shape[-2:])
        d_top = GpuCorrMM(self.border_mode,
                          self.subsample,
                          self.filter_dilation)(bottom, weights)
        d_height_width = (
            theano.gradient.DisconnectedType()(),
            ) * 2 if len(inp) == 4 else ()
        return (d_weights, d_top) + d_height_width

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0]]  # no connection to height, width


