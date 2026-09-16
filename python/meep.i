/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */

%module meep

%import "config.h"

%{
#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_2_UNICODE

/*
 * In C++ we can use a scoped variable to acquire the GIL and then auto release
 * on leaving scope, making our code a bit cleaner.
 *
 * SWIG_PYTHON_THREAD_SCOPED_BLOCK is a macro that SWIG automatically generates
 * wrapping a class using an RAII pattern to automatically acquire/release
 * the GIL. See the generated meep-python.cxx for details.
 *
 * We could instead just explicitly call SWIG_PYTHON_THREAD_BEGIN_BLOCK and
 * SWIG_PYTHON_THREAD_END_BLOCK everywhere - but this is error prone since we
 * have to ensure that SWIG_PYTHON_THREAD_END_BLOCK is called before every
 * return statement in a method.
 *
 * NOTE: This wont work with plain-old C.
 */
#define SWIG_PYTHON_THREAD_SCOPED_BLOCK   SWIG_PYTHON_THREAD_BEGIN_BLOCK

/* this #define from Python's structmember.h, used by swig, conflicts with meep.hpp */
#undef READONLY

#include <complex>
#include <string>

#include "config.h"
#include "meep/vec.hpp"
#include "meep.hpp"
#include "gpu_backend_internal.hpp"
#include "meep/mympi.hpp"
#include "ctl-math.h"
#include "ctlgeom.h"
#include "meepgeom.hpp"
#include "meep-python.hpp"

namespace meep {
    size_t dft_chunks_Ntotal(dft_chunk *dft_chunks, size_t *my_start);
    typedef std::complex<double> (*amplitude_function)(const vec &);
}

#ifdef HAVE_MPB
#include "mpb.h"

namespace meep {
    struct eigenmode_data {
        maxwell_data *mdata;
        scalar_complex *fft_data_H, *fft_data_E;
        evectmatrix H;
        int n[3];
        double s[3];
        double Gk[3];
        vec center;
        amplitude_function amp_func;
        int band_num;
        double frequency;
        double group_velocity;
    };
}
#else
namespace meep {
    struct eigenmode_data {};
}
#endif

using namespace meep;
using namespace meep_geom;

extern boolean point_in_objectp(vector3 p, GEOMETRIC_OBJECT o);
extern boolean point_in_periodic_objectp(vector3 p, GEOMETRIC_OBJECT o);
void display_geometric_object_info(int indentby, GEOMETRIC_OBJECT o);

%}

%ignore meep::eigenmode_data::mdata;
%ignore meep::eigenmode_data::fft_data_H;
%ignore meep::eigenmode_data::fft_data_E;
%ignore meep::eigenmode_data::H;

%include "numpy.i"
%include "std_vector.i"

%init %{
  import_array();
%}

%{
typedef struct {
    PyObject *func;
    int num_components;
} py_field_func_data;


#include "typemap_utils.cpp"

static PyObject *py_source_time_object() {
    // Return value: Borrowed reference
    static PyObject *source_time_object = NULL;
    if (source_time_object == NULL) {
        PyObject *source_mod = PyImport_ImportModule("meep.source");
        source_time_object = PyObject_GetAttrString(source_mod, "SourceTime");
        Py_XDECREF(source_mod);
    }
    return source_time_object;
}

static PyObject *py_meep_src_time_object() {
    // Return value: Borrowed reference
    static PyObject *src_time = NULL;
    if (src_time == NULL) {
        PyObject *meep_mod = PyImport_ImportModule("meep");
        src_time = PyObject_GetAttrString(meep_mod, "src_time");
        Py_XDECREF(meep_mod);
    }
    return src_time;
}

static double py_callback_wrap(const meep::vec &v) {
    SWIG_PYTHON_THREAD_SCOPED_BLOCK;
    PyObject *pyv = vec2py(v);
    PyObject *pyret = PyObject_CallFunctionObjArgs(py_callback, pyv, NULL);
    Py_DECREF(pyv);
    if (!pyret) { abort_with_stack_trace(); }
    double ret = PyFloat_AsDouble(pyret);
    Py_DECREF(pyret);
    return ret;
}

static std::complex<double> py_amp_func_wrap(const meep::vec &v) {
    SWIG_PYTHON_THREAD_SCOPED_BLOCK;
    PyObject *pyv = vec2py(v);
    PyObject *pyret = PyObject_CallFunctionObjArgs(py_amp_func, pyv, NULL);
    Py_DECREF(pyv);
    if (!pyret) { abort_with_stack_trace(); }
    double real = PyComplex_RealAsDouble(pyret);
    double imag = PyComplex_ImagAsDouble(pyret);
    std::complex<double> ret(real, imag);
    Py_DECREF(pyret);
    return ret;
}

static std::complex<double> py_field_func_wrap(const std::complex<meep::realnum> *fields,
                                               const meep::vec &loc,
                                               void *data_) {
    SWIG_PYTHON_THREAD_SCOPED_BLOCK;
    PyObject *pyv = vec2py(loc);

    py_field_func_data *data = (py_field_func_data *)data_;
    int len = data->num_components;

    PyObject *py_args = PyTuple_New(len + 1);
    // Increment here because PyTuple_SetItem steals a reference
    Py_INCREF(pyv);
    PyTuple_SetItem(py_args, 0, pyv);

    for (Py_ssize_t i = 1; i < len + 1; i++) {
        PyObject *cmplx = PyComplex_FromDoubles(fields[i - 1].real(), fields[i - 1].imag());
        PyTuple_SetItem(py_args, i, cmplx);
    }

    PyObject *pyret = PyObject_CallObject(data->func, py_args);

    if (!pyret) { abort_with_stack_trace(); }

    double real = PyComplex_RealAsDouble(pyret);
    double imag = PyComplex_ImagAsDouble(pyret);
    std::complex<double> ret(real, imag);
    Py_DECREF(pyv);
    Py_DECREF(pyret);
    Py_DECREF(py_args);
    return ret;
}

static meep::vec py_kpoint_func_wrap(double freq, int mode, void *user_data) {
    SWIG_PYTHON_THREAD_SCOPED_BLOCK;
    PyObject *py_freq = PyFloat_FromDouble(freq);
    PyObject *py_mode = PyInteger_FromLong(mode);

    PyObject *py_result = PyObject_CallFunctionObjArgs((PyObject*)user_data, py_freq, py_mode, NULL);

    meep::vec result;

    if (!py_result) {
        PyErr_PrintEx(0);
        result = meep::vec(0, 0, 0);
    } else {
        vector3 v3;
        if (!pyv3_to_v3(py_result, &v3)) {
            PyErr_PrintEx(0);
            result = meep::vec(0, 0, 0);
        } else {
            result = meep::vec(v3.x, v3.y, v3.z);
        }
        Py_XDECREF(py_result);
    }

    Py_DECREF(py_freq);
    Py_DECREF(py_mode);
    return result;
}

static void _do_master_printf(const char* stream_name, const char* text) {
    SWIG_PYTHON_THREAD_SCOPED_BLOCK;
    PyObject *py_stream = PySys_GetObject((char*)stream_name); // arg is non-const on Python2

    Py_XDECREF(PyObject_CallMethod(py_stream, "write", "(s)", text));
    Py_XDECREF(PyObject_CallMethod(py_stream, "flush", NULL));
}

void py_master_printf_wrap(const char *s) {
    _do_master_printf("stdout", s);
}

void py_master_printf_stderr_wrap(const char *s) {
    _do_master_printf("stderr", s);
}

void set_ctl_printf_callback(void (*callback)(const char *s)) {
#if HAVE_CTL_PRINTF_CALLBACK
  ctl_printf_callback = callback;
#else
  (void)callback;
#endif
}

void set_mpb_printf_callback(void (*callback)(const char *s)) {
#if HAVE_MPB_PRINTF_CALLBACK
  mpb_printf_callback = callback;
#else
  (void)callback;
#endif
}

static int pyabsorber_to_absorber(PyObject *py_absorber, meep_geom::absorber *a) {

    if (!get_attr_dbl(py_absorber, &a->thickness, "thickness") ||
        !get_attr_int(py_absorber, &a->direction, "direction") ||
        !get_attr_int(py_absorber, &a->side, "side") ||
        !get_attr_dbl(py_absorber, &a->R_asymptotic, "R_asymptotic") ||
        !get_attr_dbl(py_absorber, &a->mean_stretch, "mean_stretch")) {

        return 0;
    }

    PyObject *py_pml_profile_func = PyObject_GetAttrString(py_absorber, "pml_profile");

    if (!py_pml_profile_func) {
         PyErr_SetString(PyExc_AttributeError, "Class attribute 'pml_profile' is None");
         return 0;
    }

    a->pml_profile_data = py_pml_profile_func;

    return 1;
}

// Wrapper for Python PML profile function
double py_pml_profile(double u, void *f) {
    SWIG_PYTHON_THREAD_SCOPED_BLOCK;
    PyObject *func = (PyObject *)f;
    PyObject *d = PyFloat_FromDouble(u);

    PyObject *pyret = PyObject_CallFunctionObjArgs(func, d, NULL);

    if (!pyret) { abort_with_stack_trace(); }

    double ret = PyFloat_AsDouble(pyret);
    Py_XDECREF(pyret);
    Py_XDECREF(d);
    return ret;
}

PyObject *py_do_harminv(PyObject *vals, double dt, double f_min, double f_max, int maxbands,
                     double spectral_density, double Q_thresh, double rel_err_thresh,
                     double err_thresh, double rel_amp_thresh, double amp_thresh) {
    // Return value: New reference

    std::complex<double> *amp = new std::complex<double>[maxbands];
    double *freq_re = new double[maxbands];
    double *freq_im = new double[maxbands];
    double *freq_err = new double[maxbands];

    Py_ssize_t n = PyList_Size(vals);
    std::complex<double> *items = new std::complex<double>[n];

    for(int i = 0; i < n; i++) {
        Py_complex py_c = PyComplex_AsCComplex(PyList_GetItem(vals, i));
        std::complex<double> c(py_c.real, py_c.imag);
        items[i] = c;
    }

    maxbands = do_harminv(items, n, dt, f_min, f_max, maxbands, amp,
                          freq_re, freq_im, freq_err, spectral_density, Q_thresh,
                          rel_err_thresh, err_thresh, rel_amp_thresh, amp_thresh);

    PyObject *res = PyList_New(maxbands);

    for(int i = 0; i < maxbands; i++) {
        Py_complex pyfreq = {freq_re[i], freq_im[i]};
        Py_complex pyamp = {amp[i].real(), amp[i].imag()};
        Py_complex pyfreq_err = {freq_err[i], 0};

        PyObject *pyobj = Py_BuildValue("(DDD)", &pyfreq, &pyamp, &pyfreq_err);
        PyList_SetItem(res, i, pyobj);
    }

    delete[] freq_err;
    delete[] freq_im;
    delete[] freq_re;
    delete[] amp;
    delete[] items;

    return res;
}

// Wrapper around meep::dft_near2far::farfield
PyObject *_get_farfield(meep::dft_near2far *f, const meep::vec & v, double greencyl_tol) {
    // Return value: New reference
    Py_ssize_t len = f->freq.size() * 6;
    PyObject *res = PyList_New(len);

    std::complex<double> *ff_arr = f->farfield(v, greencyl_tol);

    for (Py_ssize_t i = 0; i < len; i++) {
        PyList_SetItem(res, i, PyComplex_FromDoubles(ff_arr[i].real(), ff_arr[i].imag()));
    }

    delete[] ff_arr;

    return res;
}

// Batched arbitrary-point wrapper used by adjoint Near2Far objectives and
// angular far-field scans.  The result is point-major (Npoint,Nfreq,6).
PyObject *_get_farfields_points(meep::dft_near2far *f,
                                const std::vector<meep::vec> &points,
                                double greencyl_tol) {
    npy_intp dims[3] = {
        static_cast<npy_intp>(points.size()),
        static_cast<npy_intp>(f->freq.size()),
        6,
    };
    PyObject *result = PyArray_SimpleNew(3, dims, NPY_CDOUBLE);
    if (!result) return nullptr;
    try {
        std::complex<double> *fields =
            f->farfields(points.empty() ? nullptr : points.data(),
                         points.size(), greencyl_tol);
        const size_t count = points.size() * f->freq.size() * 6;
        if (count != 0)
            memcpy(PyArray_DATA((PyArrayObject *)result), fields,
                   count * sizeof(std::complex<double>));
        delete[] fields;
    }
    catch (...) {
        Py_DECREF(result);
        throw;
    }
    return result;
}

// Wrapper around meep::dft_near2far::get_farfields_array
PyObject *_get_farfields_array(meep::dft_near2far *n2f, const meep::volume &where,
                               double resolution, double greencyl_tol) {
    // Return value: New reference
    size_t dims[4] = {1, 1, 1, 1};
    int rank = 0;
    size_t N = 1;

    double *EH = n2f->get_farfields_array(where, rank, dims, N, resolution, greencyl_tol);

    if (!EH) return PyArray_SimpleNew(0, 0, NPY_CDOUBLE);

    // frequencies are the last dimension
    if (n2f->freq.size() > 1) dims[rank++] = n2f->freq.size();

    // Additional rank to store all 12 E/H x/y/z r/i arrays.
    rank++;
    npy_intp *arr_dims = new npy_intp[rank];
    arr_dims[0] = 12;
    for (int i = 1; i < rank; ++i) {
        arr_dims[i] = dims[i - 1];
    }

    PyObject *py_arr = PyArray_SimpleNew(rank, arr_dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*)py_arr), EH, sizeof(double) * 2 * N * 6 * n2f->freq.size());

    delete[] arr_dims;
    delete[] EH;
    return py_arr;

}

// Wrapper around meep::dft_ldos::ldos
PyObject *_dft_ldos_ldos(meep::dft_ldos *f) {
    // Return value: New reference
    Py_ssize_t len = f->freq.size();
    PyObject *res = PyList_New(len);

    double *tmp = f->ldos();

    for (Py_ssize_t i = 0; i < len; i++) {
        PyList_SetItem(res, i, PyFloat_FromDouble(tmp[i]));
    }

    delete[] tmp;

    return res;
}

// Wrapper around meep::dft_ldos_F
PyObject *_dft_ldos_F(meep::dft_ldos *f) {
    // Return value: New reference
    Py_ssize_t len = f->freq.size();
    PyObject *res = PyList_New(len);

    std::complex<double> *tmp = f->F();

    for (Py_ssize_t i = 0; i < len; i++) {
        PyList_SetItem(res, i, PyComplex_FromDoubles(tmp[i].real(), tmp[i].imag()));
    }

    delete[] tmp;

    return res;
}

// Wrapper arond meep::dft_ldos_J
PyObject *_dft_ldos_J(meep::dft_ldos *f) {
    // Return value: New reference
    Py_ssize_t len = f->freq.size();
    PyObject *res = PyList_New(len);

    std::complex<double> *tmp = f->J();

    for (Py_ssize_t i = 0; i < len; i++) {
        PyList_SetItem(res, i, PyComplex_FromDoubles(tmp[i].real(), tmp[i].imag()));
    }

    delete[] tmp;

    return res;
}

/* This is a wrapper function to fool SWIG...since our list constructor
   takes ownership of the next pointer, we have to make sure that SWIG
   does not garbage-collect volume_list objects.  We do
   this by wrapping a "helper" function around the constructor which
   does not have the %newobject SWIG attribute.   Note that we then
   need to deallocate the list explicitly in Python. */
meep::volume_list *make_volume_list(const meep::volume &v, int c,
                                    std::complex<double> weight,
                                    meep::volume_list *next) {

    return new meep::volume_list(v, c, weight, next);
}

template<typename dft_type>
PyObject *_get_dft_array(meep::fields *f, dft_type dft, meep::component c, int num_freq) {
    // Return value: New reference
    int rank;
    size_t dims[3];
    std::complex<meep::realnum> *dft_arr = f->get_dft_array(dft, c, num_freq, &rank, dims);

    int npy_type = sizeof(meep::realnum) == sizeof(float) ? NPY_CFLOAT : NPY_CDOUBLE;

    if (dft_arr == NULL) { // this can happen e.g. if component c vanishes by symmetry
         PyObject *py_arr = PyArray_SimpleNew(0, 0, npy_type);
         std::complex<meep::realnum> zero(0, 0);
         memcpy(PyArray_DATA((PyArrayObject*)py_arr), &zero, sizeof(std::complex<meep::realnum>));
         return py_arr;
    }

    if (rank == 0) { // singleton results
         PyObject *py_arr = PyArray_SimpleNew(0, 0, npy_type);
         memcpy(PyArray_DATA((PyArrayObject*)py_arr), dft_arr, sizeof(std::complex<meep::realnum>));
         delete[] dft_arr;
         return py_arr;
    }

    size_t length = 1;
    npy_intp *arr_dims = new npy_intp[rank];
    for (int i = 0; i < rank; ++i) {
         arr_dims[i] = dims[i];       // implicit size_t -> int cast, presumed safe for individual array dimensions
         length *= dims[i];
    }

    PyObject *py_arr = PyArray_SimpleNew(rank, arr_dims, sizeof(meep::realnum) == sizeof(float) ? NPY_CFLOAT : NPY_CDOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*)py_arr), dft_arr, sizeof(std::complex<meep::realnum>) * length);
    delete[] dft_arr;
    if (arr_dims) delete[] arr_dims;

    return py_arr;
}

size_t _get_dft_data_size(meep::dft_chunk *dc) {
    size_t istart;
    return meep::dft_chunks_Ntotal(dc, &istart) / 2;
}

void _get_dft_data(meep::dft_chunk *dc, std::complex<double> *cdata, int size) {
    size_t istart;
    size_t n = meep::dft_chunks_Ntotal(dc, &istart) / 2;
    istart /= 2;

    if (n != (size_t)size) {
        meep::abort("Total dft_chunks size does not agree with size allocated for output array.\n");
    }

    for (meep::dft_chunk *cur = dc; cur; cur = cur->next_in_dft) {
        meep::gpu::detail::sync_resident_cache_for_owner(cur->fc);
        size_t Nchunk = cur->N * cur->omega.size();
        for (size_t i = 0; i < Nchunk; ++i) {
            cdata[i + istart] = cur->dft[i];
        }
        istart += Nchunk;
    }
}

void _load_dft_data(meep::dft_chunk *dc, std::complex<double> *cdata, int size) {
    size_t istart;
    size_t n = meep::dft_chunks_Ntotal(dc, &istart) / 2;
    istart /= 2;

    if (n != (size_t)size) {
        meep::abort("Total dft_chunks size does not agree with size allocated for output array.\n");
    }

    for (meep::dft_chunk *cur = dc; cur; cur = cur->next_in_dft) {
        // The caller is replacing this monitor's canonical host data. Drop
        // only the corresponding output mirror so a device-authoritative
        // value cannot overwrite or hide the loaded array.
        meep::gpu::detail::discard_resident_mirror_for_owner(cur->fc, cur->dft);
        size_t Nchunk = cur->N * cur->omega.size();
        for (size_t i = 0; i < Nchunk; ++i) {
            cur->dft[i] = cdata[i + istart];
        }
        istart += Nchunk;
    }
}

struct kpoint_list {
    meep::vec *kpoints;
    size_t n;
    meep::vec *kdom;
    size_t num_bands;
};

kpoint_list get_eigenmode_coefficients_and_kpoints(meep::fields *f, meep::dft_flux *flux, const meep::volume &eig_vol,
                                                   int *bands, int num_bands, int parity, double eig_resolution,
                                                   double eigensolver_tol, std::complex<double> *coeffs,
                                                   double *vgrp, meep::kpoint_func user_kpoint_func,
                                                   void *user_kpoint_data, double *cscale, meep::direction d) {

    size_t num_kpoints = num_bands * flux->freq.size();
    meep::vec *kpoints = new meep::vec[num_kpoints];
    meep::vec *kdom = new meep::vec[num_kpoints];

    f->get_eigenmode_coefficients(*flux, eig_vol, bands, num_bands, parity, eig_resolution, eigensolver_tol,
                                  coeffs, vgrp, user_kpoint_func, user_kpoint_data, kpoints, kdom, cscale, d,
                                  NULL, &flux->eigenmode_cache,
                                  &flux->eigenmode_cache_dispersive, &flux->eigenmode_cache_frequency);

    kpoint_list res = {kpoints, num_kpoints, kdom, num_kpoints};

    return res;
}

kpoint_list get_eigenmode_coefficients_and_kpoints(meep::fields *f, meep::dft_flux *flux, const meep::volume &eig_vol,
                                                   meep::diffractedplanewave dp, int parity, double eig_resolution,
                                                   double eigensolver_tol, std::complex<double> *coeffs,
                                                   double *vgrp, meep::kpoint_func user_kpoint_func,
                                                   void *user_kpoint_data, double *cscale, meep::direction d) {

    size_t num_kpoints = flux->freq.size();
    meep::vec *kpoints = new meep::vec[num_kpoints];
    meep::vec *kdom = new meep::vec[num_kpoints];
    f->get_eigenmode_coefficients(*flux, eig_vol, NULL, 1, parity, eig_resolution, eigensolver_tol,
                                  coeffs, vgrp, user_kpoint_func, user_kpoint_data, kpoints, kdom, cscale, d,
                                  &dp, &flux->eigenmode_cache,
                                  &flux->eigenmode_cache_dispersive, &flux->eigenmode_cache_frequency);

    kpoint_list res = {kpoints, num_kpoints, kdom, num_kpoints};

    return res;
}

PyObject *_get_array_slice_dimensions(meep::fields *f, const meep::volume &where, size_t dims[3],
                                      bool collapse_empty_dimensions, bool snap_empty_dimensions,
                                      meep::component cgrid = Centered, PyObject *min_max_loc = NULL) {
    // Return value: New reference
    meep::direction dirs[3] = {meep::X, meep::X, meep::X};

    meep::vec min_max_loc_vec[2];
    meep::vec* min_max_loc_vec_ptr = min_max_loc_vec;
    if (!min_max_loc) min_max_loc_vec_ptr = NULL;

    int rank = f->get_array_slice_dimensions(where, dims, dirs, collapse_empty_dimensions, snap_empty_dimensions, min_max_loc_vec_ptr, 0, cgrid);

    PyObject *py_dirs = PyList_New(3);
    for (Py_ssize_t i = 0; i < 3; ++i) {
        PyList_SetItem(py_dirs, i, PyInteger_FromLong(static_cast<int>(dirs[i])));
    }

    if (min_max_loc){
        PyObject * py_min = vec2py(min_max_loc_vec[0],true);
        PyObject * py_max = vec2py(min_max_loc_vec[1],true);
        PyList_Append(min_max_loc, py_min);
        PyList_Append(min_max_loc, py_max);
        Py_DECREF(py_min);
        Py_DECREF(py_max);
    }

    PyObject *rval = Py_BuildValue("(iO)", rank, py_dirs);
    Py_DECREF(py_dirs);
    return rval;
}

#ifdef HAVE_MPB
meep::eigenmode_data *_get_eigenmode(meep::fields *f, double frequency, meep::direction d, const meep::volume where,
                                     const meep::volume eig_vol, int band_num, const meep::vec &_kpoint,
                                     bool match_frequency, int parity, double resolution, double eigensolver_tol,
                                     double kdom[3]) {

    void *data = f->get_eigenmode(frequency, d, where, eig_vol, band_num, _kpoint, match_frequency,
                                  parity, resolution, eigensolver_tol, kdom);
    return (meep::eigenmode_data *)data;
}

meep::eigenmode_data *_get_eigenmode_dp(meep::fields *f, double frequency, meep::direction d,
                                        const meep::volume where, const meep::volume eig_vol,
                                        meep::diffractedplanewave dp, const meep::vec &_kpoint,
                                        bool match_frequency, int parity, double resolution,
                                        double eigensolver_tol, double kdom[3]) {

    void *data = f->get_eigenmode(frequency, d, where, eig_vol, 1, _kpoint, match_frequency,
                                  parity, resolution, eigensolver_tol, kdom, NULL, &dp);
    return (meep::eigenmode_data *)data;
}

PyObject *_get_eigenmode_Gk(meep::eigenmode_data *emdata) {
    SWIG_PYTHON_THREAD_SCOPED_BLOCK;
    // Return value: New reference
    PyObject *v3_class = py_vector3_object();
    PyObject *args = Py_BuildValue("(ddd)", emdata->Gk[0], emdata->Gk[1], emdata->Gk[2]);
    PyObject *result = PyObject_Call(v3_class, args, NULL);
    Py_DECREF(args);
    return result;
}

#else
void _get_eigenmode(meep::fields *f, double frequency, meep::direction d, const meep::volume where,
                    const meep::volume eig_vol, int band_num, const meep::vec &_kpoint,
                    bool match_frequency, int parity, double resolution, double eigensolver_tol,
                    double kdom[3]) {
    (void) f; (void) frequency; (void) d; (void) where; (void) eig_vol; (void) band_num; (void) _kpoint;
    (void) match_frequency; (void) parity; (void) resolution; (void) eigensolver_tol;
    (void) kdom;
    meep::abort("Must compile Meep with MPB for get_eigenmode");
}

void _get_eigenmode_dp(meep::fields *f, double frequency, meep::direction d,
                       const meep::volume where, const meep::volume eig_vol,
                       meep::diffractedplanewave dp, const meep::vec &_kpoint,
                       bool match_frequency, int parity, double resolution,
                       double eigensolver_tol, double kdom[3]) {
    (void) f; (void) frequency; (void) d; (void) where; (void) eig_vol; (void) dp; (void) _kpoint;
    (void) match_frequency; (void) parity; (void) resolution; (void) eigensolver_tol;
    (void) kdom;
    meep::abort("Must compile Meep with MPB for get_eigenmode");
}
#endif
%}

/*
 * These methods extensively use the Python C api (especially allocation) and
 * hence need to hold the GIL (acquire/release) for key parts of their
 * implementaion/code. Instead, disable threading for these methods by default.
 *
 * TODO: If any of these methods are expensive, we can explicitly allow threads
 * for the expensive blocks of code in these methods.
 */
%feature("nothreadallow") _dft_ldos_J;
%feature("nothreadallow") _dft_ldos_F;
%feature("nothreadallow") _dft_ldos_ldos;
%feature("nothreadallow") _get_farfields_array;
%feature("nothreadallow") _get_farfield;
%feature("nothreadallow") _get_farfields_points;
%feature("nothreadallow") py_do_harminv;
%feature("nothreadallow") _get_array_slice_dimensions;
%feature("nothreadallow") _get_gradient;
%feature("nothreadallow") _get_dft_array;

%numpy_typemaps(std::complex<double>, NPY_CDOUBLE, int);
%numpy_typemaps(std::complex<double>, NPY_CDOUBLE, size_t);

%apply (std::complex<double> *INPLACE_ARRAY1, int DIM1) {(std::complex<double> *cdata, int size)};

// add_volume_source
%apply (std::complex<double> *INPLACE_ARRAY3, size_t DIM1, size_t DIM2, size_t DIM3) {
    (std::complex<double> *arr, size_t dim1, size_t dim2, size_t dim3)
};

// This is necessary so that SWIG wraps py_pml_profile as a SWIG function
// pointer object instead of as a built-in function
%constant double py_pml_profile(double u, void *f);
%ignore py_pml_profile;
double py_pml_profile(double u, void *f);

%constant void py_master_printf_wrap(const char *s);
%constant void py_master_printf_stderr_wrap(const char *s);
void set_ctl_printf_callback(void (*callback)(const char *s));
void set_mpb_printf_callback(void (*callback)(const char *s));

PyObject *py_do_harminv(PyObject *vals, double dt, double f_min, double f_max, int maxbands,
                     double spectral_density, double Q_thresh, double rel_err_thresh,
                     double err_thresh, double rel_amp_thresh, double amp_thresh);

PyObject *_get_farfield(meep::dft_near2far *f, const meep::vec & v, double greencyl_tol);
PyObject *_get_farfields_points(meep::dft_near2far *f, const std::vector<meep::vec> &points, double greencyl_tol);
PyObject *_get_farfields_array(meep::dft_near2far *n2f, const meep::volume &where, double resolution, double greencyl_tol);
PyObject *_dft_ldos_ldos(meep::dft_ldos *f);
PyObject *_dft_ldos_F(meep::dft_ldos *f);
PyObject *_dft_ldos_J(meep::dft_ldos *f);
template<typename dft_type>
PyObject *_get_dft_array(meep::fields *f, dft_type dft, meep::component c, int num_freq);
size_t _get_dft_data_size(meep::dft_chunk *dc);
void _get_dft_data(meep::dft_chunk *dc, std::complex<double> *cdata, int size);
void _load_dft_data(meep::dft_chunk *dc, std::complex<double> *cdata, int size);
meep::volume_list *make_volume_list(const meep::volume &v, int c,
                                    std::complex<double> weight,
                                    meep::volume_list *next);

// Typemap suite for get_eigenmode_coefficients_and_kpoints

%typemap(out) kpoint_list {

    PyObject *py_kpoints = PyList_New($1.n);
    PyObject *py_kdom = PyList_New($1.num_bands);

    for (size_t i = 0; i < $1.n; ++i) {
        PyList_SetItem(py_kpoints, i, vec2py($1.kpoints[i], true));
    }
    for (size_t i = 0; i < $1.num_bands; ++i) {
        PyList_SetItem(py_kdom, i, vec2py($1.kdom[i], true));
    }

    $result = Py_BuildValue("(O,O)", py_kpoints, py_kdom);

    Py_DECREF(py_kpoints);
    Py_DECREF(py_kdom);
    delete[] $1.kpoints;
    delete[] $1.kdom;
}

// Typemap suite for do_harminv

%typecheck(SWIG_TYPECHECK_POINTER) PyObject *vals {
    $1 = PyList_Check($input);
}

// Typemap suite for double func(meep::vec &)

%typemap(in) double (*)(const meep::vec &) {
  if ($input == Py_None) {
    $1 = NULL;
    py_callback = NULL;
  } else {
    $1 = py_callback_wrap;
    py_callback = $input;
    Py_INCREF(py_callback);
  }
}

%typemap(freearg) double (*)(const meep::vec &) {
  Py_XDECREF(py_callback);
  py_callback = NULL;
}

%typecheck(SWIG_TYPECHECK_POINTER) double (*)(const meep::vec &) {
  $1 = PyCallable_Check($input) || $input == Py_None;
}

// Typemap suite for amplitude function

%typecheck(SWIG_TYPECHECK_POINTER) std::complex<double> (*)(const meep::vec &) {
  $1 = $input == Py_None || PyCallable_Check($input);
}

%typemap(in) std::complex<double> (*)(const meep::vec &) {
     if ($input != Py_None) {
          $1 = py_amp_func_wrap;
          py_amp_func = $input;
          Py_INCREF(py_amp_func);
     }
     else
          $1 = NULL;
}

%typemap(freearg) std::complex<double> (*)(const meep::vec &) {
    Py_XDECREF(py_amp_func);
    py_amp_func = NULL;
}

// Typemap suite for vector3

%typecheck (SWIG_TYPECHECK_POINTER) vector3 {
    $1 = PyObject_IsInstance($input, py_vector3_object());
}

%typemap(in) vector3 {
    if(!pyv3_to_v3($input, &$1)) {
        SWIG_fail;
    }
}

// Typemap suite for GEOMETRIC_OBJECT

%typemap(in) GEOMETRIC_OBJECT {
    if(!py_gobj_to_gobj($input, &$1)) {
        SWIG_fail;
    }
}

%typemap(arginit) GEOMETRIC_OBJECT {
    $1.material = NULL;
}

%typemap(freearg) GEOMETRIC_OBJECT {
    if ($1.material) {
        material_free((material_data *)$1.material);
        geometric_object_destroy($1);
    }
}

%typemap(out) geometric_object {
    $result = gobj_to_py_obj(&$1);

    if (!$result) {
        SWIG_fail;
    }
}

// Typemap suite for boolean

%typemap(out) boolean {
    $result = PyBool_FromLong($1);
}

// Typemap suite for geometric_object_list

%typecheck(SWIG_TYPECHECK_POINTER) geometric_object_list {
    $1 = PyList_Check($input);
}

%typemap(in) geometric_object_list {
    if(!py_list_to_gobj_list($input, &$1)) {
        SWIG_fail;
    }
}

%typemap(in) geometric_object_list* (geometric_object_list temp){
    if(!py_list_to_gobj_list($input, &temp)) {
        SWIG_fail;
    }
    $1 = &temp;
}

%typemap(arginit) geometric_object_list {
    $1.num_items = 0;
    $1.items = NULL;
}

%typemap(freearg) geometric_object_list {
    gobj_list_freearg(&$1);
}

%typemap(freearg) geometric_object_list* {
    gobj_list_freearg($1);
}

%typemap(out) geometric_object_list {
    $result = gobj_list_to_py_list(&$1);

    if (!$result) {
        SWIG_fail;
    }
}

// Typemap suite for susceptibility_list

%typecheck(SWIG_TYPECHECK_POINTER) susceptibility_list {
    $1 = PyList_Check($input);
}

%typemap(in) susceptibility_list {
    if(!py_list_to_susceptibility_list($input, &$1)) {
        SWIG_fail;
    }
}


//--------------------------------------------------
// typemaps needed for material grid
//--------------------------------------------------

%inline %{
void _get_gradient(PyObject *grad, double scalegrad,
                   meep::dft_fields *fields_a_0, meep::dft_fields *fields_a_1, meep::dft_fields *fields_a_2,
                   meep::dft_fields *fields_f_0, meep::dft_fields *fields_f_1, meep::dft_fields *fields_f_2,
                   meep::grid_volume *grid_volume, PyObject *frequencies,
                   meep_geom::geom_epsilon *geps, double fd_step) {

    // clean the gradient array
    PyArrayObject *pao_grad = (PyArrayObject *)grad;
    if (!PyArray_Check(pao_grad)) meep::abort("grad parameter must be numpy array.");
    if (!PyArray_ISCARRAY(pao_grad)) meep::abort("Numpy grad array must be C-style contiguous.");
    if (PyArray_NDIM(pao_grad) !=2) {meep::abort("Numpy grad array must have 2 dimensions.");}
    double *grad_c = (double *)PyArray_DATA(pao_grad);
    npy_intp ng = PyArray_DIMS(pao_grad)[1]; // number of design parameters

    // clean the adjoint fields object
    std::vector<meep::dft_fields *> adjoint_fields = {fields_a_0,fields_a_1,fields_a_2};

    // clean the forward fields object
    std::vector<meep::dft_fields *> forward_fields = {fields_f_0,fields_f_1,fields_f_2};

    // clean the frequencies array
    PyArrayObject *pao_freqs = (PyArrayObject *)frequencies;
    if (!PyArray_Check(pao_freqs)) meep::abort("frequencies parameter must be numpy array.");
    if (!PyArray_ISCARRAY(pao_freqs)) meep::abort("Numpy fields array must be C-style contiguous.");
    double *frequencies_c = (double *)PyArray_DATA(pao_freqs);
    npy_intp nf = PyArray_DIMS(pao_freqs)[0];
    if (PyArray_DIMS(pao_grad)[0] != nf) meep::abort("Numpy grad array is allocated for %td frequencies; it should be allocated for %td.",PyArray_DIMS(pao_grad)[0],nf);

    // calculate the gradient
    meep_geom::material_grids_addgradient(grad_c,ng,nf,adjoint_fields,forward_fields,frequencies_c,scalegrad,*grid_volume,geps,fd_step);

}
%}

//--------------------------------------------------
// end typemaps needed for material grid
//--------------------------------------------------

// Typemap suite for sources

%typecheck(SWIG_TYPECHECK_POINTER) const meep::src_time & {
    int py_source_time = PyObject_IsInstance($input, py_source_time_object());
    int swig_src_time = PyObject_IsInstance($input, py_meep_src_time_object());

    $1 = py_source_time || swig_src_time;
}

%typemap(in) const meep::src_time & {
    PyObject *swig_obj = NULL;
    void *tmp_ptr = 0;
    int tmp_res = 0;

    if(PyObject_IsInstance($input, py_source_time_object())) {
        swig_obj = PyObject_GetAttrString($input, "swigobj");
    } else if(PyObject_IsInstance($input, py_meep_src_time_object())) {
        swig_obj = $input;
        Py_XINCREF(swig_obj);
    } else {
      meep::abort("Expected a meep.source.SourceTime or a meep.src_time\n");
    }

    tmp_res = SWIG_ConvertPtr(swig_obj, &tmp_ptr, $1_descriptor, 0);
    Py_XDECREF(swig_obj);

    if(!SWIG_IsOK(tmp_res)) {
        SWIG_exception_fail(SWIG_ArgError(tmp_res), "Couldn't convert Python object to meep::src_time");
    }
    $1 = reinterpret_cast<meep::src_time *>(tmp_ptr);

}

// Typemap suite for boundary_region

%typecheck(SWIG_TYPECHECK_POINTER) void *pml_profile_data {
    $1 = PyCallable_Check($input);
}

%typemap(in) void *pml_profile_data {
    $1 = (void*)$input;
}

// Typemap suite for dft_flux

%typemap(out) double* flux {
    int size = arg1->freq.size();
    $result = PyList_New(size);
    for(int i = 0; i < size; i++) {
        PyList_SetItem($result, i, PyFloat_FromDouble($1[i]));
    }

    delete[] $1;
}

%typemap(out) std::vector<std::complex<double>> complexflux {
    size_t size = $1.size();
    $result = PyList_New(size);
    for(size_t i = 0; i < size; i++) {
        PyList_SetItem($result, i, PyComplex_FromDoubles(real($1[i]), imag($1[i])));
    }
}

// Typemap suite for dft_force

%typemap(out) double* force {
    int size = arg1->freq.size();
    $result = PyList_New(size);
    for(int i = 0; i < size; i++) {
        PyList_SetItem($result, i, PyFloat_FromDouble($1[i]));
    }

    delete[] $1;
}

// Typemap suite for material_type

%typecheck(SWIG_TYPECHECK_POINTER) material_type {
    int py_material = PyObject_IsInstance($input, py_material_object());
    int user_material = PyFunction_Check($input);
    int file_material = IsPyString($input);
    int numpy_material = PyArray_Check($input);

    $1 = py_material || user_material || file_material || numpy_material;
}

%typemap(in) material_type {
    if(!pymaterial_to_material($input, &$1)) {
        SWIG_fail;
    }
}

%typemap(arginit) material_type {
    $1 = NULL;
}

%typemap(freearg) material_type {
    if ($1) {
        material_free($1);
    }
}

// For some reason SWIG needs the namespaced version too
%apply material_type { meep_geom::material_type };

// Typemap for numpy array passed as double* (used by get_array_metadata,
// get_epsilon_grid, get_eigenmode_coefficients, near2far, etc.)

%typecheck(SWIG_TYPECHECK_POINTER, fragment="NumPy_Fragments") double* xtics {
    $1 = is_array($input);
}
%typemap(in, fragment="NumPy_Macros") double* xtics {
    $1 = (double *)array_data($input);
}
%apply double* xtics {
     double* ytics, double* ztics, double* weights,
     double* vgrp, double* cscale, double* farpt_list
};

// Typemap suite for array_slice

%typecheck(SWIG_TYPECHECK_POINTER, fragment="NumPy_Fragments") size_t dims[3] {
    $1 = is_array($input);
}

%typemap(in, fragment="NumPy_Macros") size_t dims[3] {
    $1 = (size_t *)array_data($input);
}

%typecheck(SWIG_TYPECHECK_POINTER, fragment="NumPy_Fragments") meep::realnum* slice {
    $1 = is_array($input);
}

%typemap(in, fragment="NumPy_Macros") meep::realnum* slice {
    $1 = (meep::realnum *)array_data($input);
}

%typecheck(SWIG_TYPECHECK_POINTER, fragment="NumPy_Fragments") std::complex<meep::realnum>* slice {
    $1 = is_array($input);
}

%typemap(in) std::complex<meep::realnum>* slice {
    $1 = (std::complex<meep::realnum> *)array_data($input);
}

%typecheck(SWIG_TYPECHECK_POINTER) meep::component {
    $1 = PyInteger_Check($input) && PyInteger_AsLong($input) < 100;
}

%typemap(in) meep::component {
    $1 = static_cast<meep::component>(PyInteger_AsLong($input));
}

%typecheck(SWIG_TYPECHECK_POINTER) meep::derived_component {
    $1 = PyInteger_Check($input) && PyInteger_AsLong($input) >= 100;
}

%typemap(in) meep::derived_component {
    $1 = static_cast<meep::derived_component>(PyInteger_AsLong($input));
}

%typecheck(SWIG_TYPECHECK_POINTER) PyObject *min_max_loc {
    $1 = PyList_Check($input);
}

%apply int INPLACE_ARRAY1[ANY] { int [3] };
%apply double INPLACE_ARRAY1[ANY] { double [3] };

// Typemap for numpy array passed as std::complex<double>* (used by
// get_epsilon_grid, solve_cw, get_eigenmode_coefficients, adjoint, etc.)

%typecheck(SWIG_TYPECHECK_POINTER, fragment="NumPy_Fragments") std::complex<double>* grid_vals {
    $1 = is_array($input);
}
%typemap(in, fragment="NumPy_Macros") std::complex<double>* grid_vals {
    $1 = (std::complex<double> *)array_data($input);
}
%apply std::complex<double>* grid_vals {
     std::complex<double>* eigfreq, std::complex<double>* coeffs,
     std::complex<double>* dJ, std::complex<double>* amp_arr
};

// typemaps for diffractedplanewave

%typecheck(SWIG_TYPECHECK_POINTER, fragment="NumPy_Fragments") double axis[3] {
    $1 = is_array($input);
}

%typemap(in) double axis[3] {
     $1 = (double *)array_data($input);
}

// typemaps for gaussianbeam

%typecheck(SWIG_TYPECHECK_POINTER, fragment="NumPy_Fragments") std::complex<double> E0[3] {
    $1 = is_array($input);
}

%typemap(in) std::complex<double> E0[3] {
     $1 = (std::complex<double> *)array_data($input);
}

// typemap for get_eigenmode_coefficients bands array
%apply (int *IN_ARRAY1, int DIM1) {(int *bands, int num_bands)};


// typemaps for add_dft_fields

%apply (const double* IN_ARRAY1, size_t DIM1) {(const double* freq, size_t Nfreq)}

%typecheck(SWIG_TYPECHECK_POINTER) const volume where {
    int py_material = PyObject_IsInstance($input, py_volume_object());
    $1 = py_material;
}

%typecheck(SWIG_TYPECHECK_POINTER) meep::component *components {
    $1 = PyList_Check($input);
}

%typemap(in) (meep::component *components, int num_components) {
    if (!PyList_Check($input)) {
        meep::abort("Expected a list");
    }
    $2 = PyList_Size($input);
    $1 = new meep::component[$2];

    for (Py_ssize_t i = 0; i < $2; i++) {
        $1[i] = (meep::component)PyInteger_AsLong(PyList_GetItem($input, i));
    }
}

%typemap(freearg) (meep::component *components, int num_components) {
    delete[] $1;
}
//--------------------------------------------------
// end typemaps for add_dft_fields
//--------------------------------------------------

// typemap suite for field functions

%typecheck(SWIG_TYPECHECK_POINTER) (int num_fields, const meep::component *components,
                                    meep::field_function fun, void *fun_data_) {
    $1 = PySequence_Check($input) &&
         PySequence_Check(PyList_GetItem($input, 0)) &&
         PyCallable_Check(PyList_GetItem($input, 1));
}
%typemap(in) (int num_fields, const meep::component *components, meep::field_function fun, void *fun_data_)
    (py_field_func_data tmp_data) {

    if (!PySequence_Check($input)) {
        meep::abort("Expected a sequence");
    }

    PyObject *cs = PyList_GetItem($input, 0);

    if (!PySequence_Check(cs)) {
        meep::abort("Expected first item in list to be a list");
    }

    PyObject *func = PyList_GetItem($input, 1);

    if (!PyCallable_Check(func)) {
        meep::abort("Expected a function");
    }

    $1 = PyList_Size(cs);
    $2 = new meep::component[$1];

    for (Py_ssize_t i = 0; i < $1; i++) {
        $2[i] = (meep::component)PyInteger_AsLong(PyList_GetItem(cs, i));
    }

    $3 = py_field_func_wrap;

    tmp_data.num_components = $1;
    tmp_data.func = func;
    Py_INCREF(tmp_data.func);
    $4 = &tmp_data;
}

%typemap(freearg) (int num_fields, const meep::component *components, meep::field_function fun, void *fun_data_) {
    delete[] $2;
    Py_XDECREF(tmp_data$argnum.func);
}

// integrate2
%typecheck(SWIG_TYPECHECK_POINTER) (int num_fields1, const meep::component *components1, int num_fields2,
                                    const meep::component *components2, meep::field_function integrand,
                                    void *integrand_data_) {
    $1 = PySequence_Check($input) &&
         PySequence_Check(PyList_GetItem($input, 0)) &&
         PySequence_Check(PyList_GetItem($input, 1)) &&
         PyCallable_Check(PyList_GetItem($input, 2));
}

%typemap(in) (int num_fields1, const meep::component *components1, int num_fields2,
              const meep::component *components2, meep::field_function integrand,
              void *integrand_data_) (py_field_func_data data) {

    if (!PySequence_Check($input)) {
        meep::abort("Expected a sequence");
    }

    PyObject *cs1 = PyList_GetItem($input, 0);

    if (!PySequence_Check(cs1)) {
        meep::abort("Expected 1st item in list to be a sequence");
    }

    PyObject *cs2 = PyList_GetItem($input, 1);

    if (!PySequence_Check(cs2)) {
        meep::abort("Expected 2nd item in list to be a sequence");
    }

    PyObject *func = PyList_GetItem($input, 2);

    if (!PyCallable_Check(func)) {
        meep::abort("Expected 3rd item in list to be a function");
    }

    $1 = PyList_Size(cs1);
    $3 = PyList_Size(cs2);

    $2 = new meep::component[$1];
    $4 = new meep::component[$3];

    for (Py_ssize_t i = 0; i < $1; i++) {
        $2[i] = (meep::component)PyInteger_AsLong(PyList_GetItem(cs1, i));
    }

    for (Py_ssize_t i = 0; i < $3; i++) {
        $4[i] = (meep::component)PyInteger_AsLong(PyList_GetItem(cs2, i));
    }

    $5 = py_field_func_wrap;

    data.num_components = $1 + $3;
    data.func = func;
    Py_INCREF(func);
    $6 = &data;
}

%typemap(freearg) (int num_fields1, const meep::component *components1, int num_fields2,
                   const meep::component *components2, meep::field_function integrand, void *integrand_data_) {
    delete[] $2;
    delete[] $4;
    Py_XDECREF(data$argnum.func);
}

// Typemap suite for absorber_list

%typecheck(SWIG_TYPECHECK_POINTER) meep_geom::absorber_list {
    $1 = PySequence_Check($input);
}

%typemap(in) meep_geom::absorber_list {

    Py_ssize_t len = PyList_Size($input);

    if (len == 0) {
        $1 = 0;
    } else {
        $1 = create_absorber_list();

        for (Py_ssize_t i = 0; i < len; i++) {
            absorber a;
            PyObject *py_absorber = PyList_GetItem($input, i);

            if (!pyabsorber_to_absorber(py_absorber, &a)) {
                SWIG_fail;
            }

            add_absorbing_layer($1, a.thickness, a.direction, a.side,
                                a.R_asymptotic, a.mean_stretch, py_pml_profile,
                                a.pml_profile_data);
            Py_DECREF((PyObject *)a.pml_profile_data);
        }
    }
}

%typemap(arginit) meep_geom::absorber_list {
    $1 = NULL;
}

%typemap(freearg) meep_geom::absorber_list {
    destroy_absorber_list($1);
}

// Typemap suite for material_type_list

%typecheck(SWIG_TYPECHECK_POINTER) material_type_list {
    $1 = PySequence_Check($input);
}

%typemap(in) material_type_list {
    Py_ssize_t len = PyList_Size($input);

    if (len == 0) {
        $1 = material_type_list();
    } else {
        material_type_list mtl;
        mtl.num_items = len;
        mtl.items = new material_type[len];
        for (Py_ssize_t i = 0; i < len; i++) {
            PyObject *py_material = PyList_GetItem($input, i);
            if (!pymaterial_to_material(py_material, &mtl.items[i])) {
                SWIG_fail;
            }
        }
        $1 = mtl;
    }
}

%typemap(arginit) material_type_list {
    $1.num_items = 0;
    $1.items = NULL;
}

%typemap(freearg) material_type_list {
    if ($1.num_items != 0) {
        for (int i = 0; i < $1.num_items; i++) {
            material_free($1.items[i]);
        }
    }
    delete[] $1.items;
}

// For some reason SWIG needs the namespaced version too
%apply material_type_list { meep_geom::material_type_list };

// Typemap suite for kpoint_func

%typecheck(SWIG_TYPECHECK_POINTER) (meep::kpoint_func user_kpoint_func, void *user_kpoint_data) {
    $1 = PyCallable_Check($input) || $input == Py_None;
}

%typemap(in) (meep::kpoint_func user_kpoint_func, void *user_kpoint_data) {
    if ($input == Py_None) {
        $1 = NULL;
        $2 = NULL;
    }
    else {
        $1 = py_kpoint_func_wrap;
        $2 = (void*)$input;
    }
}

%apply double *flux {
    double *electric,
    double *magnetic,
    double *total
};

// dJ, amp_arr, fpart_list typemaps are covered by %apply directives above

%exception {
#ifdef MEEP_SWIG_PYTHON_DEBUG
  // NOTE: You can do fancier things like timing the calls and using that
  // to track the most expensive calls etc.
  master_printf("**SWIG**: $symname\n");
#endif
  try {
    $action
  } catch (std::runtime_error &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    SWIG_fail;
  }
}

// typemaps for binary_partition

%typecheck (SWIG_TYPECHECK_POINTER) meep::binary_partition * {
    $1 = PyObject_IsInstance($input, py_binary_partition_object());
}

%typemap(in) meep::binary_partition * (std::unique_ptr<meep::binary_partition> temp){
  temp = py_bp_to_bp($input);
  $1 = temp.get();
}

%typemap(out) const meep::binary_partition * {
  $result = bp_to_py_bp($1);
}

%typemap(arginit) meep::binary_partition * {
    $1 = NULL;
}


// typemaps for timing data

%typemap(out) std::unordered_map<meep::time_sink, std::vector<double>, std::hash<int> > {
  PyObject *out_dict = PyDict_New();
  for (const auto& ts_vec : $1) {
    const std::vector<double>& timing_vector = ts_vec.second;
    PyObject *res = PyList_New(timing_vector.size());
    for (size_t i = 0; i < timing_vector.size(); ++i) {
      PyList_SetItem(res, i, PyFloat_FromDouble(timing_vector[i]));
    }
    PyObject *key = PyInteger_FromLong(static_cast<int>(ts_vec.first));
    PyDict_SetItem(out_dict, key, res);

    Py_DECREF(key);
    Py_DECREF(res);
  }
  $result = out_dict;
}


// Tells Python to take ownership of the h5file* this function returns so that
// it gets garbage collected and the file gets closed.
%newobject meep::fields::open_h5file;

%newobject meep::make_output_directory;
%newobject _get_eigenmode;
%newobject _get_eigenmode_dp;

%rename(_vec) meep::vec::vec;
%rename(_dft_ldos) meep::dft_ldos::dft_ldos;

// Rename python builtins
%rename(br_apply) meep::boundary_region::apply;
%rename(_is) meep::dft_chunk::is;
%rename(Meep_None) meep::None;

// Operator renaming
%rename(boundary_region_assign) meep::boundary_region::operator=;

%rename(get_field_from_comp) meep::fields::get_field(component, const vec &) const;

%feature("python:cdefaultargs") meep::fields::add_eigenmode_source;

%feature("immutable") meep::fields_chunk::connections;
%feature("immutable") meep::fields_chunk::num_connections;

%ignore susceptibility_equal;
%ignore susceptibility_list_equal;
%ignore medium_struct_equal;
%ignore material_gc;
%ignore material_type_equal;
%ignore is_variable;
%ignore is_file;
%ignore is_medium;
%ignore is_metal;
%ignore meep::all_in_or_out;
%ignore meep::all_connect_phases;
%ignore meep::choose_chunkdivision;
%ignore meep::comms_key;
%ignore meep::comms_key_hash_fn;
%ignore meep::comms_manager;
%ignore meep::comms_operation;
%ignore meep::comms_sequence;
%ignore meep::comms_overlap_statistics;
%ignore meep::create_comms_manager;
%ignore meep::comms_supports_cuda_device_buffers;
%ignore meep::comms_start_cuda_device_receives;
%ignore meep::comms_start_cuda_device_sends;
%ignore meep::reset_comms_overlap_statistics;
%ignore meep::get_comms_overlap_statistics;
%ignore meep::comms_physical_message_count;
%ignore meep::comms_finish;
%ignore meep::fields::get_time_spent_on;
%ignore meep::fields::times_spent;
%ignore meep::fields::was_working_on;
%ignore meep::fields::with_timing_scope;
%ignore meep::fields::working_on;
%ignore meep::fields_chunk;
%ignore meep::infinity;
%ignore meep::timing_scope;

%ignore std::vector<meep::volume>::vector(size_type);
%ignore std::vector<meep::volume>::resize;
%ignore std::vector<meep_geom::dft_data>::vector(size_type);
%ignore std::vector<meep_geom::dft_data>::resize;

%ignore meep_geom::set_materials_from_geom_epsilon;

// template instantiations
%template(get_dft_flux_array) _get_dft_array<meep::dft_flux>;
%template(get_dft_fields_array) _get_dft_array<meep::dft_fields>;
%template(get_dft_force_array) _get_dft_array<meep::dft_force>;
%template(get_dft_near2far_array) _get_dft_array<meep::dft_near2far>;

%template(FragmentStatsVector) std::vector<meep_geom::fragment_stats>;
%template(DftDataVector) std::vector<meep_geom::dft_data>;
%template(VolumeVector) std::vector<meep::volume>;
%template(GridVolumeVector) std::vector<meep::grid_volume>;
%template(IntVector) std::vector<int>;
%template(Size_t_Vector) std::vector<size_t>;
%template(DoubleVector) std::vector<double>;
%template(VecVector) std::vector<meep::vec>;

// use NumPy arrays for returning common std::vector types:
%typemap(out) std::vector<double> {
    npy_intp vec_len = (npy_intp) $1.size();
    $result = PyArray_SimpleNew(1, &vec_len, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*) $result), &$1[0], vec_len * sizeof(double));
}
%typemap(out) std::vector<int> {
    npy_intp vec_len = (npy_intp) $1.size();
    $result = PyArray_SimpleNew(1, &vec_len, NPY_INT);
    memcpy(PyArray_DATA((PyArrayObject*) $result), &$1[0], vec_len * sizeof(int));
}

%include "vec.i"
%include "meep.hpp"
%include "meep/mympi.hpp"
%include "meepgeom.hpp"
%include "meep-python.hpp"

%include "typemaps.i"
%template(near_src_data) std::vector<meep::sourcedata>;

%include "std_complex.i"
%template(ComplexVector) std::vector<std::complex<double> >;

%typemap(out) std::vector<std::complex<double> > {
    npy_intp vec_len = (npy_intp) $1.size();
    $result = PyArray_SimpleNew(1, &vec_len, NPY_COMPLEX128);
    memcpy(PyArray_DATA((PyArrayObject*) $result), &$1[0], vec_len * sizeof(double) * 2);
}

/*
 * Keep the public Python GPU surface independent of CUDA headers and Python
 * extension types. The same module therefore imports in CPU-only builds.
 */
%inline %{
struct gpu_python_statistics {
    unsigned long long runtime_availability_probes;
    unsigned long long runtime_device_enumerations;
    unsigned long long runtime_device_selections;
    unsigned long long cpu_curl_calls;
    unsigned long long cpu_curl_points;
    unsigned long long cuda_curl_calls;
    unsigned long long cuda_curl_points;
    unsigned long long host_to_device_bytes;
    unsigned long long device_to_host_bytes;
    unsigned long long host_to_device_bytes_avoided;
    unsigned long long device_to_host_bytes_avoided;
    unsigned long long device_buffer_allocations;
    unsigned long long device_buffer_reuses;
    unsigned long long live_resident_device_buffers;
    unsigned long long cpu_update_eh_calls;
    unsigned long long cpu_update_eh_points;
    unsigned long long cuda_update_eh_calls;
    unsigned long long cuda_update_eh_points;
    unsigned long long cpu_polarization_calls;
    unsigned long long cpu_polarization_points;
    unsigned long long cuda_polarization_calls;
    unsigned long long cuda_polarization_points;
    unsigned long long cpu_source_calls;
    unsigned long long cpu_source_points;
    unsigned long long cuda_source_calls;
    unsigned long long cuda_source_points;
    unsigned long long cpu_boundary_calls;
    unsigned long long cpu_boundary_points;
    unsigned long long cuda_boundary_calls;
    unsigned long long cuda_boundary_points;
    unsigned long long cpu_dft_calls;
    unsigned long long cpu_dft_points;
    unsigned long long cuda_dft_calls;
    unsigned long long cuda_dft_points;
    unsigned long long dft_batch_calls;
    unsigned long long dft_submitted_updates;
    unsigned long long dft_phase_preparation_launches;
    unsigned long long dft_phase_reuses;
    unsigned long long dft_update_kernel_launches;
    unsigned long long dft_maximum_batch_size;
    unsigned long long dft_multi_monitor_automatic_checks;
    unsigned long long dft_multi_monitor_automatic_selected;
    unsigned long long dft_multi_monitor_automatic_rejected;
    unsigned long long dft_multi_monitor_forced_batches;
    unsigned long long dft_multi_monitor_batched_updates;
    unsigned long long dft_multi_monitor_unbatched_updates;
    unsigned long long dft_multi_monitor_plan_uploads;
    unsigned long long dft_multi_monitor_plan_reuses;
    unsigned long long dft_multi_monitor_metadata_host_to_device_bytes;
    unsigned long long cpu_dft_reduction_calls;
    unsigned long long cpu_dft_reduction_pairs;
    unsigned long long cpu_dft_reduction_terms;
    unsigned long long cuda_dft_reduction_calls;
    unsigned long long cuda_dft_reduction_pairs;
    unsigned long long cuda_dft_reduction_terms;
    unsigned long long cuda_dft_reduction_descriptor_uploads;
    unsigned long long cuda_dft_reduction_plan_reuses;
    unsigned long long cuda_dft_reduction_kernel_launches;
    unsigned long long cuda_dft_reduction_result_device_to_host_bytes;
    unsigned long long dft_reduction_full_dft_device_to_host_bytes_avoided;
    unsigned long long dft_reduction_mpi_allreduce_calls;
    unsigned long long dft_reduction_mpi_allreduce_bytes;
    unsigned long long cpu_dft_array_materialization_calls;
    unsigned long long cpu_dft_array_materialization_points;
    unsigned long long cuda_dft_array_materialization_calls;
    unsigned long long cuda_dft_array_materialization_points;
    unsigned long long host_synthetic_material_array_calls;
    unsigned long long host_synthetic_material_array_points;
    unsigned long long cpu_dft_output_calls;
    unsigned long long cpu_dft_output_points;
    unsigned long long cuda_dft_output_calls;
    unsigned long long cuda_dft_output_points;
    unsigned long long cuda_dft_output_staging_calls;
    unsigned long long cuda_dft_output_staging_points;
    unsigned long long cuda_dft_output_staging_frequencies;
    unsigned long long cuda_dft_output_staging_descriptor_uploads;
    unsigned long long cuda_dft_output_staging_plan_reuses;
    unsigned long long cuda_dft_output_staging_kernel_launches;
    unsigned long long cuda_dft_output_staging_result_device_to_host_bytes;
    unsigned long long cuda_dft_output_staging_full_dft_device_to_host_bytes_avoided;
    unsigned long long cuda_dft_output_staging_workspace_ceiling_bytes;
    unsigned long long cpu_eigenmode_overlap_calls;
    unsigned long long cpu_eigenmode_overlap_terms;
    unsigned long long cuda_eigenmode_overlap_calls;
    unsigned long long cuda_eigenmode_overlap_terms;
    unsigned long long cuda_eigenmode_mode_flux_calls;
    unsigned long long cuda_eigenmode_mode_mode_calls;
    unsigned long long cuda_eigenmode_submitted_pairs;
    unsigned long long cuda_eigenmode_descriptor_uploads;
    unsigned long long cuda_eigenmode_plan_reuses;
    unsigned long long cuda_eigenmode_kernel_launches;
    unsigned long long cuda_eigenmode_result_device_to_host_bytes;
    unsigned long long eigenmode_full_dft_device_to_host_bytes_avoided;
    unsigned long long host_mode_profile_sampling_calls;
    unsigned long long host_mode_profile_sampling_points;
    unsigned long long eigenmode_zero_rank_channels_skipped;
    unsigned long long host_mode_profile_host_to_device_bytes;
    unsigned long long eigenmode_mpi_allreduce_calls;
    unsigned long long eigenmode_mpi_allreduce_bytes;
    unsigned long long cuda_dft_materialization_kernel_launches;
    unsigned long long cuda_dft_materialization_result_device_to_host_bytes;
    unsigned long long dft_materialization_full_dft_device_to_host_bytes_avoided;
    unsigned long long dft_array_mpi_allreduce_calls;
    unsigned long long dft_array_mpi_allreduce_bytes;
    unsigned long long cpu_dft_checkpoint_save_calls;
    unsigned long long cpu_dft_checkpoint_save_values;
    unsigned long long cuda_dft_checkpoint_save_calls;
    unsigned long long cuda_dft_checkpoint_save_values;
    unsigned long long cuda_dft_checkpoint_save_device_to_host_bytes;
    unsigned long long cuda_dft_checkpoint_save_full_cache_device_to_host_bytes_avoided;
    unsigned long long cpu_dft_checkpoint_load_calls;
    unsigned long long cpu_dft_checkpoint_load_values;
    unsigned long long cuda_dft_checkpoint_load_calls;
    unsigned long long cuda_dft_checkpoint_load_values;
    unsigned long long cuda_dft_checkpoint_load_host_to_device_bytes;
    unsigned long long cpu_dft_scale_calls;
    unsigned long long cpu_dft_scale_values;
    unsigned long long cuda_dft_scale_calls;
    unsigned long long cuda_dft_scale_values;
    unsigned long long cuda_dft_scale_kernel_launches;
    unsigned long long cuda_dft_scale_host_to_device_bytes;
    unsigned long long cpu_ldos_reduction_calls;
    unsigned long long cpu_ldos_source_points;
    unsigned long long cuda_ldos_reduction_calls;
    unsigned long long cuda_ldos_submitted_profiles;
    unsigned long long cuda_ldos_source_points;
    unsigned long long cuda_ldos_descriptor_uploads;
    unsigned long long cuda_ldos_kernel_launches;
    unsigned long long cuda_ldos_result_device_to_host_bytes;
    unsigned long long ldos_full_field_device_to_host_bytes_avoided;
    unsigned long long cpu_near2far_transform_calls;
    unsigned long long cpu_near2far_terms;
    unsigned long long cuda_near2far_transform_calls;
    unsigned long long cuda_near2far_terms;
    unsigned long long cuda_near2far_submitted_chunks;
    unsigned long long cuda_near2far_source_points;
    unsigned long long cuda_near2far_output_points;
    unsigned long long cuda_near2far_frequencies;
    unsigned long long cuda_near2far_periodic_copies;
    unsigned long long cuda_near2far_fast_precision_calls;
    unsigned long long cuda_near2far_mixed_precision_calls;
    unsigned long long cuda_near2far_cancellation_retries;
    unsigned long long cuda_near2far_target_tiles;
    unsigned long long cuda_near2far_frequency_tiles;
    unsigned long long cuda_near2far_operation_tiles;
    unsigned long long cuda_near2far_maximum_workspace_bytes;
    unsigned long long cuda_near2far_descriptor_uploads;
    unsigned long long cuda_near2far_kernel_launches;
    unsigned long long cuda_near2far_result_device_to_host_bytes;
    unsigned long long cuda_near2far_condition_device_to_host_bytes;
    unsigned long long near2far_dft_device_to_host_bytes_avoided;
    unsigned long long near2far_mpi_allreduce_calls;
    unsigned long long near2far_mpi_allreduce_bytes;
    unsigned long long cpu_near2far_adjoint_calls;
    unsigned long long cpu_near2far_adjoint_terms;
    unsigned long long cuda_near2far_adjoint_calls;
    unsigned long long cuda_near2far_adjoint_terms;
    unsigned long long cuda_near2far_adjoint_submitted_chunks;
    unsigned long long cuda_near2far_adjoint_source_points;
    unsigned long long cuda_near2far_adjoint_far_points;
    unsigned long long cuda_near2far_adjoint_frequencies;
    unsigned long long cuda_near2far_adjoint_periodic_copies;
    unsigned long long cuda_near2far_adjoint_fast_precision_calls;
    unsigned long long cuda_near2far_adjoint_mixed_precision_calls;
    unsigned long long cuda_near2far_adjoint_cancellation_retries;
    unsigned long long cuda_near2far_adjoint_maximum_workspace_bytes;
    unsigned long long cuda_near2far_adjoint_descriptor_uploads;
    unsigned long long cuda_near2far_adjoint_kernel_launches;
    unsigned long long cuda_near2far_adjoint_host_to_device_bytes;
    unsigned long long cuda_near2far_adjoint_result_device_to_host_bytes;
    unsigned long long cuda_near2far_adjoint_condition_device_to_host_bytes;
    unsigned long long mpi_messages;
    unsigned long long mpi_scalars;
    unsigned long long cuda_aware_bytes;
    unsigned long long pinned_staging_bytes;
    unsigned long long pinned_device_to_host_bytes;
    unsigned long long pinned_host_to_device_bytes;
    unsigned long long mpi_waitsome_executions;
    unsigned long long mpi_waitall_executions;
    unsigned long long boundary_eh_overlap_checks;
    unsigned long long boundary_eh_overlap_eligible;
    unsigned long long boundary_eh_overlap_launched_h;
    unsigned long long boundary_eh_overlap_launched_e;
    unsigned long long boundary_eh_overlap_skipped_disabled;
    unsigned long long boundary_eh_overlap_skipped_unsupported_schedule;
    unsigned long long boundary_eh_overlap_skipped_no_remote;
    unsigned long long boundary_eh_overlap_skipped_cold_topology;
    unsigned long long boundary_eh_overlap_rejected;
    unsigned long long halo_curl_overlap_checks;
    unsigned long long halo_curl_overlap_eligible;
    unsigned long long halo_curl_overlap_launches;
    unsigned long long halo_curl_overlap_skipped_disabled;
    unsigned long long halo_curl_overlap_skipped_unsupported_schedule;
    unsigned long long halo_curl_overlap_skipped_no_remote;
    unsigned long long halo_curl_overlap_skipped_cold_topology;
    unsigned long long halo_curl_overlap_rejected_feature;
    unsigned long long halo_curl_overlap_rejected_small;
    unsigned long long halo_curl_overlap_full_points;
    unsigned long long halo_curl_overlap_interior_points;
    unsigned long long halo_curl_overlap_shell_points;
    unsigned long long tile_coalesced_curl_chunk_phases;
    unsigned long long tile_coalesced_curl_input_tiles;
    unsigned long long tile_coalesced_update_eh_chunk_phases;
    unsigned long long tile_coalesced_update_eh_input_tiles;
    unsigned long long phase_curl_automatic_checks;
    unsigned long long phase_curl_automatic_selected;
    unsigned long long phase_curl_automatic_rejected;
    unsigned long long phase_curl_forced_batches;
    unsigned long long phase_curl_batched_operations;
    unsigned long long phase_curl_unbatched_operations;
    unsigned long long phase_curl_replay_checks;
    unsigned long long phase_curl_replay_hits;
    unsigned long long phase_curl_replay_unready;
    unsigned long long phase_curl_replay_generation_misses;
    unsigned long long phase_curl_replay_mirror_misses;
    unsigned long long phase_update_eh_automatic_checks;
    unsigned long long phase_update_eh_automatic_selected;
    unsigned long long phase_update_eh_automatic_rejected;
    unsigned long long phase_update_eh_forced_batches;
    unsigned long long phase_update_eh_batched_operations;
    unsigned long long phase_update_eh_unbatched_operations;
};

bool _gpu_backend_compiled() {
    return meep::gpu::backend_compiled();
}

void _gpu_initialize_distributed_runtime() {
    meep::initialize_distributed_gpu_runtime();
}

void _gpu_finalize_distributed_runtime() {
    meep::finalize_distributed_gpu_runtime();
}

bool _gpu_runtime_available() {
    return meep::gpu::runtime_available(nullptr);
}

const char *_gpu_runtime_diagnostic() {
    static thread_local std::string diagnostic;
    diagnostic.clear();
    meep::gpu::runtime_available(&diagnostic);
    return diagnostic.c_str();
}

const char *_gpu_compiled_architectures() {
    static thread_local std::string architectures;
    architectures = meep::gpu::compiled_architectures();
    return architectures.c_str();
}

void _gpu_set_backend(int mode) {
    switch (mode) {
      case 0:
        meep::gpu::set_backend(meep::gpu::backend_mode::cpu);
        return;
      case 1:
        meep::gpu::set_backend(meep::gpu::backend_mode::automatic);
        return;
      case 2:
        meep::gpu::set_backend(meep::gpu::backend_mode::cuda);
        return;
      default:
        throw std::runtime_error(
            "GPU backend must be 'cpu', 'auto', or 'cuda'");
    }
}

int _gpu_requested_backend() {
    return static_cast<int>(meep::gpu::requested_backend());
}

int _gpu_active_backend() {
    return static_cast<int>(meep::gpu::active_backend());
}

const char *_gpu_backend_diagnostic() {
    static thread_local std::string diagnostic;
    diagnostic = meep::gpu::backend_diagnostic();
    return diagnostic.c_str();
}

void _gpu_select_device(int ordinal) {
    meep::gpu::select_device(ordinal);
}

int _gpu_selected_device() {
    return meep::gpu::selected_device();
}

const char *_gpu_selected_device_identifier() {
    static thread_local std::string identifier;
    identifier = meep::gpu::selected_device_identifier();
    return identifier.c_str();
}

int _gpu_device_count() {
    return static_cast<int>(meep::gpu::enumerate_devices().size());
}

int _gpu_device_integer_property(int index, int property) {
    const std::vector<meep::gpu::device_info> devices =
        meep::gpu::enumerate_devices();
    if (index < 0 || static_cast<std::size_t>(index) >= devices.size())
        throw std::runtime_error("GPU device index is out of range");
    const meep::gpu::device_info &device =
        devices[static_cast<std::size_t>(index)];
    switch (property) {
      case 0: return device.ordinal;
      case 1: return device.compute_major;
      case 2: return device.compute_minor;
      case 3: return device.multiprocessor_count;
      case 4: return device.max_threads_per_block;
      case 5: return device.compatible ? 1 : 0;
      default:
        throw std::runtime_error("unknown GPU device integer property");
    }
}

unsigned long long _gpu_device_memory(int index) {
    const std::vector<meep::gpu::device_info> devices =
        meep::gpu::enumerate_devices();
    if (index < 0 || static_cast<std::size_t>(index) >= devices.size())
        throw std::runtime_error("GPU device index is out of range");
    return static_cast<unsigned long long>(
        devices[static_cast<std::size_t>(index)].global_memory_bytes);
}

unsigned long long _gpu_device_memory_bandwidth(int index) {
    const std::vector<meep::gpu::device_info> devices =
        meep::gpu::enumerate_devices();
    if (index < 0 || static_cast<std::size_t>(index) >= devices.size())
        throw std::runtime_error("GPU device index is out of range");
    return static_cast<unsigned long long>(
        devices[static_cast<std::size_t>(index)]
            .memory_bandwidth_bytes_per_second);
}

const char *_gpu_device_name(int index) {
    static thread_local std::string name;
    const std::vector<meep::gpu::device_info> devices =
        meep::gpu::enumerate_devices();
    if (index < 0 || static_cast<std::size_t>(index) >= devices.size())
        throw std::runtime_error("GPU device index is out of range");
    name = devices[static_cast<std::size_t>(index)].name;
    return name.c_str();
}

const char *_gpu_device_identifier(int index) {
    static thread_local std::string identifier;
    const std::vector<meep::gpu::device_info> devices =
        meep::gpu::enumerate_devices();
    if (index < 0 || static_cast<std::size_t>(index) >= devices.size())
        throw std::runtime_error("GPU device index is out of range");
    identifier = devices[static_cast<std::size_t>(index)].identifier;
    return identifier.c_str();
}

gpu_python_statistics _gpu_statistics() {
    const meep::gpu::runtime_touch_statistics runtime =
        meep::gpu::get_runtime_touch_statistics();
    const meep::gpu::dispatch_statistics dispatch =
        meep::gpu::get_dispatch_statistics();
    const meep::gpu::resident_statistics resident =
        meep::gpu::get_resident_statistics();
    const meep::gpu::field_update_statistics fields =
        meep::gpu::get_field_update_statistics();
    const meep::gpu::polarization_statistics polarizations =
        meep::gpu::get_polarization_statistics();
    const meep::gpu::source_statistics sources =
        meep::gpu::get_source_statistics();
    const meep::gpu::boundary_statistics boundaries =
        meep::gpu::get_boundary_statistics();
    const meep::gpu::dft_statistics dfts =
        meep::gpu::get_dft_statistics();
    const meep::gpu::dft_batch_statistics dft_batches =
        meep::gpu::get_dft_batch_statistics();
    const meep::gpu::dft_reduction_statistics dft_reductions =
        meep::gpu::get_dft_reduction_statistics();
    const meep::gpu::dft_materialization_statistics dft_materializations =
        meep::gpu::get_dft_materialization_statistics();
    const meep::gpu::dft_checkpoint_statistics dft_checkpoints =
        meep::gpu::get_dft_checkpoint_statistics();
    const meep::gpu::dft_scale_statistics dft_scales =
        meep::gpu::get_dft_scale_statistics();
    const meep::gpu::eigenmode_overlap_statistics eigenmode_overlaps =
        meep::gpu::get_eigenmode_overlap_statistics();
    const meep::gpu::ldos_statistics ldos =
        meep::gpu::get_ldos_statistics();
    const meep::gpu::near2far_statistics near2far =
        meep::gpu::get_near2far_statistics();
    const meep::gpu::multi_gpu_statistics multi_gpu =
        meep::gpu::get_multi_gpu_statistics();
    const meep::gpu::mpi_completion_statistics mpi_completion =
        meep::gpu::get_mpi_completion_statistics();
    const meep::gpu::boundary_eh_overlap_statistics boundary_eh_overlap =
        meep::gpu::get_boundary_eh_overlap_statistics();
    const meep::gpu::halo_curl_overlap_statistics halo_curl_overlap =
        meep::gpu::get_halo_curl_overlap_statistics();
    const meep::gpu::tile_coalescing_statistics tile_coalescing =
        meep::gpu::get_tile_coalescing_statistics();
    const meep::gpu::phase_batch_policy_statistics phase_batch_policy =
        meep::gpu::get_phase_batch_policy_statistics();
    const meep::gpu::curl_phase_replay_statistics curl_phase_replay =
        meep::gpu::get_curl_phase_replay_statistics();
    const gpu_python_statistics result = {
        runtime.availability_probes,
        runtime.device_enumerations,
        runtime.device_selections,
        dispatch.cpu_curl_calls,
        dispatch.cpu_curl_points,
        dispatch.cuda_curl_calls,
        dispatch.cuda_curl_points,
        dispatch.host_to_device_bytes,
        dispatch.device_to_host_bytes,
        resident.host_to_device_bytes_avoided,
        resident.device_to_host_bytes_avoided,
        resident.device_buffer_allocations,
        resident.device_buffer_reuses,
        meep::gpu::get_live_resident_device_buffers(),
        fields.cpu_update_eh_calls,
        fields.cpu_update_eh_points,
        fields.cuda_update_eh_calls,
        fields.cuda_update_eh_points,
        polarizations.cpu_update_calls,
        polarizations.cpu_update_points,
        polarizations.cuda_update_calls,
        polarizations.cuda_update_points,
        sources.cpu_update_calls,
        sources.cpu_update_points,
        sources.cuda_update_calls,
        sources.cuda_update_points,
        boundaries.cpu_update_calls,
        boundaries.cpu_update_points,
        boundaries.cuda_update_calls,
        boundaries.cuda_update_points,
        dfts.cpu_update_calls,
        dfts.cpu_update_points,
        dfts.cuda_update_calls,
        dfts.cuda_update_points,
        dft_batches.batch_calls,
        dft_batches.submitted_updates,
        dft_batches.phase_preparation_launches,
        dft_batches.phase_reuses,
        dft_batches.update_kernel_launches,
        dft_batches.maximum_batch_size,
        dft_batches.multi_monitor_automatic_checks,
        dft_batches.multi_monitor_automatic_selected,
        dft_batches.multi_monitor_automatic_rejected,
        dft_batches.multi_monitor_forced_batches,
        dft_batches.multi_monitor_batched_updates,
        dft_batches.multi_monitor_unbatched_updates,
        dft_batches.multi_monitor_plan_uploads,
        dft_batches.multi_monitor_plan_reuses,
        dft_batches.multi_monitor_metadata_host_to_device_bytes,
        dft_reductions.cpu_reduction_calls,
        dft_reductions.cpu_submitted_pairs,
        dft_reductions.cpu_point_frequency_terms,
        dft_reductions.cuda_reduction_calls,
        dft_reductions.cuda_submitted_pairs,
        dft_reductions.cuda_point_frequency_terms,
        dft_reductions.cuda_descriptor_uploads,
        dft_reductions.cuda_plan_reuses,
        dft_reductions.cuda_kernel_launches,
        dft_reductions.cuda_result_device_to_host_bytes,
        dft_reductions.full_dft_device_to_host_bytes_avoided,
        dft_reductions.mpi_allreduce_calls,
        dft_reductions.mpi_allreduce_bytes,
        dft_materializations.cpu_array_calls,
        dft_materializations.cpu_array_points,
        dft_materializations.cuda_array_calls,
        dft_materializations.cuda_array_points,
        dft_materializations.host_synthetic_material_array_calls,
        dft_materializations.host_synthetic_material_array_points,
        dft_materializations.cpu_output_calls,
        dft_materializations.cpu_output_points,
        dft_materializations.cuda_output_calls,
        dft_materializations.cuda_output_points,
        dft_materializations.cuda_output_staging_calls,
        dft_materializations.cuda_output_staging_points,
        dft_materializations.cuda_output_staging_frequencies,
        dft_materializations.cuda_output_staging_descriptor_uploads,
        dft_materializations.cuda_output_staging_plan_reuses,
        dft_materializations.cuda_output_staging_kernel_launches,
        dft_materializations.cuda_output_staging_result_device_to_host_bytes,
        dft_materializations.cuda_output_staging_full_dft_device_to_host_bytes_avoided,
        dft_materializations.cuda_output_staging_workspace_ceiling_bytes,
        eigenmode_overlaps.cpu_overlap_calls,
        eigenmode_overlaps.cpu_overlap_terms,
        eigenmode_overlaps.cuda_overlap_calls,
        eigenmode_overlaps.cuda_overlap_terms,
        eigenmode_overlaps.cuda_mode_flux_calls,
        eigenmode_overlaps.cuda_mode_mode_calls,
        eigenmode_overlaps.cuda_submitted_pairs,
        eigenmode_overlaps.cuda_descriptor_uploads,
        eigenmode_overlaps.cuda_plan_reuses,
        eigenmode_overlaps.cuda_kernel_launches,
        eigenmode_overlaps.cuda_result_device_to_host_bytes,
        eigenmode_overlaps.full_dft_device_to_host_bytes_avoided,
        eigenmode_overlaps.host_mode_profile_sampling_calls,
        eigenmode_overlaps.host_mode_profile_sampling_points,
        eigenmode_overlaps.zero_rank_channels_skipped,
        eigenmode_overlaps.host_mode_profile_host_to_device_bytes,
        eigenmode_overlaps.mpi_allreduce_calls,
        eigenmode_overlaps.mpi_allreduce_bytes,
        dft_materializations.cuda_kernel_launches,
        dft_materializations.cuda_result_device_to_host_bytes,
        dft_materializations.full_dft_device_to_host_bytes_avoided,
        dft_materializations.array_mpi_allreduce_calls,
        dft_materializations.array_mpi_allreduce_bytes,
        dft_checkpoints.cpu_save_dataset_calls,
        dft_checkpoints.cpu_save_values,
        dft_checkpoints.cuda_save_dataset_calls,
        dft_checkpoints.cuda_save_values,
        dft_checkpoints.cuda_save_device_to_host_bytes,
        dft_checkpoints.cuda_save_full_cache_device_to_host_bytes_avoided,
        dft_checkpoints.cpu_load_dataset_calls,
        dft_checkpoints.cpu_load_values,
        dft_checkpoints.cuda_load_dataset_calls,
        dft_checkpoints.cuda_load_values,
        dft_checkpoints.cuda_load_host_to_device_bytes,
        dft_scales.cpu_scale_calls,
        dft_scales.cpu_scale_values,
        dft_scales.cuda_scale_calls,
        dft_scales.cuda_scale_values,
        dft_scales.cuda_scale_kernel_launches,
        dft_scales.cuda_scale_host_to_device_bytes,
        ldos.cpu_reduction_calls,
        ldos.cpu_source_points,
        ldos.cuda_reduction_calls,
        ldos.cuda_submitted_profiles,
        ldos.cuda_source_points,
        ldos.cuda_descriptor_uploads,
        ldos.cuda_kernel_launches,
        ldos.cuda_result_device_to_host_bytes,
        ldos.full_field_device_to_host_bytes_avoided,
        near2far.cpu_transform_calls,
        near2far.cpu_terms,
        near2far.cuda_transform_calls,
        near2far.cuda_terms,
        near2far.cuda_submitted_chunks,
        near2far.cuda_source_points,
        near2far.cuda_output_points,
        near2far.cuda_frequencies,
        near2far.cuda_periodic_copies,
        near2far.cuda_fast_precision_calls,
        near2far.cuda_mixed_precision_calls,
        near2far.cuda_cancellation_retries,
        near2far.cuda_target_tiles,
        near2far.cuda_frequency_tiles,
        near2far.cuda_operation_tiles,
        near2far.cuda_maximum_workspace_bytes,
        near2far.cuda_descriptor_uploads,
        near2far.cuda_kernel_launches,
        near2far.cuda_result_device_to_host_bytes,
        near2far.cuda_condition_device_to_host_bytes,
        near2far.dft_device_to_host_bytes_avoided,
        near2far.mpi_allreduce_calls,
        near2far.mpi_allreduce_bytes,
        near2far.cpu_adjoint_calls,
        near2far.cpu_adjoint_terms,
        near2far.cuda_adjoint_calls,
        near2far.cuda_adjoint_terms,
        near2far.cuda_adjoint_submitted_chunks,
        near2far.cuda_adjoint_source_points,
        near2far.cuda_adjoint_far_points,
        near2far.cuda_adjoint_frequencies,
        near2far.cuda_adjoint_periodic_copies,
        near2far.cuda_adjoint_fast_precision_calls,
        near2far.cuda_adjoint_mixed_precision_calls,
        near2far.cuda_adjoint_cancellation_retries,
        near2far.cuda_adjoint_maximum_workspace_bytes,
        near2far.cuda_adjoint_descriptor_uploads,
        near2far.cuda_adjoint_kernel_launches,
        near2far.cuda_adjoint_host_to_device_bytes,
        near2far.cuda_adjoint_result_device_to_host_bytes,
        near2far.cuda_adjoint_condition_device_to_host_bytes,
        multi_gpu.mpi_messages,
        multi_gpu.mpi_scalars,
        multi_gpu.cuda_aware_bytes,
        multi_gpu.pinned_staging_bytes,
        multi_gpu.pinned_device_to_host_bytes,
        multi_gpu.pinned_host_to_device_bytes,
        mpi_completion.waitsome_executions,
        mpi_completion.waitall_executions,
        boundary_eh_overlap.checks,
        boundary_eh_overlap.eligible,
        boundary_eh_overlap.launched_h,
        boundary_eh_overlap.launched_e,
        boundary_eh_overlap.skipped_disabled,
        boundary_eh_overlap.skipped_unsupported_schedule,
        boundary_eh_overlap.skipped_no_remote,
        boundary_eh_overlap.skipped_cold_topology,
        boundary_eh_overlap.rejected,
        halo_curl_overlap.checks,
        halo_curl_overlap.eligible,
        halo_curl_overlap.launches,
        halo_curl_overlap.skipped_disabled,
        halo_curl_overlap.skipped_unsupported_schedule,
        halo_curl_overlap.skipped_no_remote,
        halo_curl_overlap.skipped_cold_topology,
        halo_curl_overlap.rejected_feature,
        halo_curl_overlap.rejected_small,
        halo_curl_overlap.full_points,
        halo_curl_overlap.interior_points,
        halo_curl_overlap.shell_points,
        tile_coalescing.curl_chunk_phases,
        tile_coalescing.curl_input_tiles,
        tile_coalescing.update_eh_chunk_phases,
        tile_coalescing.update_eh_input_tiles,
        phase_batch_policy.curl_automatic_checks,
        phase_batch_policy.curl_automatic_selected,
        phase_batch_policy.curl_automatic_rejected,
        phase_batch_policy.curl_forced_batches,
        phase_batch_policy.curl_batched_operations,
        phase_batch_policy.curl_unbatched_operations,
        curl_phase_replay.checks,
        curl_phase_replay.hits,
        curl_phase_replay.unready,
        curl_phase_replay.generation_misses,
        curl_phase_replay.mirror_misses,
        phase_batch_policy.update_eh_automatic_checks,
        phase_batch_policy.update_eh_automatic_selected,
        phase_batch_policy.update_eh_automatic_rejected,
        phase_batch_policy.update_eh_forced_batches,
        phase_batch_policy.update_eh_batched_operations,
        phase_batch_policy.update_eh_unbatched_operations};
    return result;
}

void _gpu_reset_statistics() {
    meep::gpu::reset_dispatch_statistics();
}
%}

struct vector3 {
    double x;
    double y;
    double z;
};

struct geom_box {
    vector3 low;
    vector3 high;
};

%rename(is_point_in_object) point_in_objectp(vector3 p, GEOMETRIC_OBJECT o);
%rename(is_point_in_periodic_object) point_in_periodic_objectp(vector3 p, GEOMETRIC_OBJECT o);

#ifdef HAVE_MPB
namespace meep {
struct eigenmode_data {
    maxwell_data *mdata;
    scalar_complex *fft_data_H, *fft_data_E;
    evectmatrix H;
    int n[3];
    double s[3];
    double Gk[3];
    vec center;
    amplitude_function amp_func;
    int band_num;
    double frequency;
    double group_velocity;
};
}

meep::eigenmode_data *_get_eigenmode(meep::fields *f, double frequency, meep::direction d, const meep::volume where,
                                     const meep::volume eig_vol, int band_num, const meep::vec &_kpoint,
                                     bool match_frequency, int parity, double resolution, double eigensolver_tol,
                                     double kdom[3]);
meep::eigenmode_data *_get_eigenmode_dp(meep::fields *f, double frequency, meep::direction d,
                                        const meep::volume where, const meep::volume eig_vol,
                                        meep::diffractedplanewave dp, const meep::vec &_kpoint,
                                        bool match_frequency, int parity, double resolution,
                                        double eigensolver_tol, double kdom[3]);
PyObject *_get_eigenmode_Gk(meep::eigenmode_data *emdata);

%extend meep::eigenmode_data {
    ~eigenmode_data() {
        meep::destroy_eigenmode_data($self);
    }
}

#else
void _get_eigenmode(meep::fields *f, double frequency, meep::direction d, const meep::volume where,
                    const meep::volume eig_vol, int band_num, const meep::vec &_kpoint,
                    bool match_frequency, int parity, double resolution, double eigensolver_tol,
                    double kdom[3]);
void _get_eigenmode_dp(meep::fields *f, double frequency, meep::direction d,
                       const meep::volume where, const meep::volume eig_vol,
                       meep::diffractedplanewave dp, const meep::vec &_kpoint,
                       bool match_frequency, int parity, double resolution,
                       double eigensolver_tol, double kdom[3]);
#endif // HAVE_MPB

%extend meep::fields {
  bool is_periodic(boundary_side side, direction dir) {
    return $self->boundaries[side][dir] == meep::Periodic;
  }
}

extern boolean point_in_objectp(vector3 p, GEOMETRIC_OBJECT o);
extern boolean point_in_periodic_objectp(vector3 p, GEOMETRIC_OBJECT o);
void display_geometric_object_info(int indentby, GEOMETRIC_OBJECT o);
kpoint_list get_eigenmode_coefficients_and_kpoints(meep::fields *f, meep::dft_flux *flux,
                                                   const meep::volume &eig_vol, int *bands, int num_bands,
                                                   int parity, double eig_resolution, double eigensolver_tol,
                                                   std::complex<double> *coeffs, double *vgrp,
                                                   meep::kpoint_func user_kpoint_func, void *user_kpoint_data,
                                                   double *cscale, meep::direction d);
kpoint_list get_eigenmode_coefficients_and_kpoints(meep::fields *f, meep::dft_flux *flux,
                                                   const meep::volume &eig_vol, meep::diffractedplanewave dp,
                                                   int parity, double eig_resolution, double eigensolver_tol,
                                                   std::complex<double> *coeffs, double *vgrp,
                                                   meep::kpoint_func user_kpoint_func, void *user_kpoint_data,
                                                   double *cscale, meep::direction d);
PyObject *_get_array_slice_dimensions(meep::fields *f, const meep::volume &where, size_t dims[3],
                                      bool collapse_empty_dimensions, bool snap_empty_dimensions,
                                      meep::component cgrid = Centered, PyObject *min_max_loc = NULL);

%ignore eps_func;
%ignore inveps_func;

%pythoncode %{
    AUTOMATIC = -1
    CYLINDRICAL = -2
    ALL = -1
    ALL_COMPONENTS = Dielectric

    # MPB definitions
    NO_PARITY = 0
    EVEN_Z = 1
    ODD_Z = 2
    EVEN_Y = 4
    ODD_Y = 8
    TE = EVEN_Z
    TM = ODD_Z
    PREV_PARITY = -1

    inf = 1.0e20

    class _GpuController:
        """Process-local CUDA backend discovery, selection, and statistics."""

        _BACKENDS = {"cpu": 0, "auto": 1, "automatic": 1, "cuda": 2}
        _BACKEND_NAMES = ("cpu", "auto", "cuda")

        @property
        def compiled(self):
            return _gpu_backend_compiled()

        @property
        def runtime_available(self):
            return _gpu_runtime_available()

        @property
        def runtime_diagnostic(self):
            return _gpu_runtime_diagnostic()

        @property
        def compiled_architectures(self):
            return _gpu_compiled_architectures()

        @property
        def requested_backend(self):
            return self._BACKEND_NAMES[_gpu_requested_backend()]

        @property
        def active_backend(self):
            return self._BACKEND_NAMES[_gpu_active_backend()]

        @property
        def backend_diagnostic(self):
            return _gpu_backend_diagnostic()

        @property
        def selected_device(self):
            return _gpu_selected_device()

        @property
        def selected_device_identifier(self):
            return _gpu_selected_device_identifier()

        def set_backend(self, mode):
            try:
                backend = self._BACKENDS[str(mode).lower()]
            except KeyError as exc:
                raise ValueError(
                    "GPU backend must be 'cpu', 'auto', or 'cuda'"
                ) from exc
            _gpu_set_backend(backend)

        def select_device(self, ordinal):
            _gpu_select_device(int(ordinal))

        def devices(self):
            # Device enumeration itself raises when the CUDA runtime has no
            # visible device (for example in a login node or restricted
            # container). Discovery is a query API, so represent that ordinary
            # state as an empty inventory while preserving the diagnostic via
            # runtime_diagnostic.
            if not self.runtime_available:
                return []
            devices = []
            for index in range(_gpu_device_count()):
                devices.append(
                    {
                        "ordinal": _gpu_device_integer_property(index, 0),
                        "compute_capability": (
                            _gpu_device_integer_property(index, 1),
                            _gpu_device_integer_property(index, 2),
                        ),
                        "multiprocessor_count": _gpu_device_integer_property(
                            index, 3
                        ),
                        "max_threads_per_block": _gpu_device_integer_property(
                            index, 4
                        ),
                        "compatible": bool(
                            _gpu_device_integer_property(index, 5)
                        ),
                        "global_memory_bytes": _gpu_device_memory(index),
                        "memory_bandwidth_bytes_per_second": (
                            _gpu_device_memory_bandwidth(index)
                        ),
                        "identifier": _gpu_device_identifier(index),
                        "name": _gpu_device_name(index),
                    }
                )
            return devices

        def reset_statistics(self):
            _gpu_reset_statistics()

        def statistics(self):
            stats = _gpu_statistics()

            def values(names):
                return {name: int(getattr(stats, name)) for name in names}

            return {
                "runtime": values(
                    (
                        "runtime_availability_probes",
                        "runtime_device_enumerations",
                        "runtime_device_selections",
                    )
                ),
                "dispatch": values(
                    (
                        "cpu_curl_calls",
                        "cpu_curl_points",
                        "cuda_curl_calls",
                        "cuda_curl_points",
                        "host_to_device_bytes",
                        "device_to_host_bytes",
                    )
                ),
                "resident": values(
                    (
                        "host_to_device_bytes_avoided",
                        "device_to_host_bytes_avoided",
                        "device_buffer_allocations",
                        "device_buffer_reuses",
                        "live_resident_device_buffers",
                    )
                ),
                "field_updates": values(
                    (
                        "cpu_update_eh_calls",
                        "cpu_update_eh_points",
                        "cuda_update_eh_calls",
                        "cuda_update_eh_points",
                    )
                ),
                "polarizations": values(
                    (
                        "cpu_polarization_calls",
                        "cpu_polarization_points",
                        "cuda_polarization_calls",
                        "cuda_polarization_points",
                    )
                ),
                "sources": values(
                    (
                        "cpu_source_calls",
                        "cpu_source_points",
                        "cuda_source_calls",
                        "cuda_source_points",
                    )
                ),
                "boundaries": values(
                    (
                        "cpu_boundary_calls",
                        "cpu_boundary_points",
                        "cuda_boundary_calls",
                        "cuda_boundary_points",
                    )
                ),
                "dfts": values(
                    (
                        "cpu_dft_calls",
                        "cpu_dft_points",
                        "cuda_dft_calls",
                        "cuda_dft_points",
                    )
                ),
                "dft_batches": values(
                    (
                        "dft_batch_calls",
                        "dft_submitted_updates",
                        "dft_phase_preparation_launches",
                        "dft_phase_reuses",
                        "dft_update_kernel_launches",
                        "dft_maximum_batch_size",
                        "dft_multi_monitor_automatic_checks",
                        "dft_multi_monitor_automatic_selected",
                        "dft_multi_monitor_automatic_rejected",
                        "dft_multi_monitor_forced_batches",
                        "dft_multi_monitor_batched_updates",
                        "dft_multi_monitor_unbatched_updates",
                        "dft_multi_monitor_plan_uploads",
                        "dft_multi_monitor_plan_reuses",
                        (
                            "dft_multi_monitor_metadata_"
                            "host_to_device_bytes"
                        ),
                    )
                ),
                "dft_reductions": values(
                    (
                        "cpu_dft_reduction_calls",
                        "cpu_dft_reduction_pairs",
                        "cpu_dft_reduction_terms",
                        "cuda_dft_reduction_calls",
                        "cuda_dft_reduction_pairs",
                        "cuda_dft_reduction_terms",
                        "cuda_dft_reduction_descriptor_uploads",
                        "cuda_dft_reduction_plan_reuses",
                        "cuda_dft_reduction_kernel_launches",
                        (
                            "cuda_dft_reduction_result_"
                            "device_to_host_bytes"
                        ),
                        (
                            "dft_reduction_full_dft_"
                            "device_to_host_bytes_avoided"
                        ),
                        "dft_reduction_mpi_allreduce_calls",
                        "dft_reduction_mpi_allreduce_bytes",
                    )
                ),
                "dft_materializations": values(
                    (
                        "cpu_dft_array_materialization_calls",
                        "cpu_dft_array_materialization_points",
                        "cuda_dft_array_materialization_calls",
                        "cuda_dft_array_materialization_points",
                        "host_synthetic_material_array_calls",
                        "host_synthetic_material_array_points",
                        "cpu_dft_output_calls",
                        "cpu_dft_output_points",
                        "cuda_dft_output_calls",
                        "cuda_dft_output_points",
                        "cuda_dft_output_staging_calls",
                        "cuda_dft_output_staging_points",
                        "cuda_dft_output_staging_frequencies",
                        "cuda_dft_output_staging_descriptor_uploads",
                        "cuda_dft_output_staging_plan_reuses",
                        "cuda_dft_output_staging_kernel_launches",
                        (
                            "cuda_dft_output_staging_result_"
                            "device_to_host_bytes"
                        ),
                        (
                            "cuda_dft_output_staging_full_dft_"
                            "device_to_host_bytes_avoided"
                        ),
                        "cuda_dft_output_staging_workspace_ceiling_bytes",
                        "cuda_dft_materialization_kernel_launches",
                        (
                            "cuda_dft_materialization_result_"
                            "device_to_host_bytes"
                        ),
                        (
                            "dft_materialization_full_dft_"
                            "device_to_host_bytes_avoided"
                        ),
                        "dft_array_mpi_allreduce_calls",
                        "dft_array_mpi_allreduce_bytes",
                    )
                ),
                "dft_checkpoints": values(
                    (
                        "cpu_dft_checkpoint_save_calls",
                        "cpu_dft_checkpoint_save_values",
                        "cuda_dft_checkpoint_save_calls",
                        "cuda_dft_checkpoint_save_values",
                        (
                            "cuda_dft_checkpoint_save_"
                            "device_to_host_bytes"
                        ),
                        (
                            "cuda_dft_checkpoint_save_full_cache_"
                            "device_to_host_bytes_avoided"
                        ),
                        "cpu_dft_checkpoint_load_calls",
                        "cpu_dft_checkpoint_load_values",
                        "cuda_dft_checkpoint_load_calls",
                        "cuda_dft_checkpoint_load_values",
                        (
                            "cuda_dft_checkpoint_load_"
                            "host_to_device_bytes"
                        ),
                    )
                ),
                "dft_scales": values(
                    (
                        "cpu_dft_scale_calls",
                        "cpu_dft_scale_values",
                        "cuda_dft_scale_calls",
                        "cuda_dft_scale_values",
                        "cuda_dft_scale_kernel_launches",
                        "cuda_dft_scale_host_to_device_bytes",
                    )
                ),
                "eigenmode_overlaps": values(
                    (
                        "cpu_eigenmode_overlap_calls",
                        "cpu_eigenmode_overlap_terms",
                        "cuda_eigenmode_overlap_calls",
                        "cuda_eigenmode_overlap_terms",
                        "cuda_eigenmode_mode_flux_calls",
                        "cuda_eigenmode_mode_mode_calls",
                        "cuda_eigenmode_submitted_pairs",
                        "cuda_eigenmode_descriptor_uploads",
                        "cuda_eigenmode_plan_reuses",
                        "cuda_eigenmode_kernel_launches",
                        "cuda_eigenmode_result_device_to_host_bytes",
                        "eigenmode_full_dft_device_to_host_bytes_avoided",
                        "host_mode_profile_sampling_calls",
                        "host_mode_profile_sampling_points",
                        "eigenmode_zero_rank_channels_skipped",
                        "host_mode_profile_host_to_device_bytes",
                        "eigenmode_mpi_allreduce_calls",
                        "eigenmode_mpi_allreduce_bytes",
                    )
                ),
                "ldos": values(
                    (
                        "cpu_ldos_reduction_calls",
                        "cpu_ldos_source_points",
                        "cuda_ldos_reduction_calls",
                        "cuda_ldos_submitted_profiles",
                        "cuda_ldos_source_points",
                        "cuda_ldos_descriptor_uploads",
                        "cuda_ldos_kernel_launches",
                        "cuda_ldos_result_device_to_host_bytes",
                        "ldos_full_field_device_to_host_bytes_avoided",
                    )
                ),
                "near2far": values(
                    (
                        "cpu_near2far_transform_calls",
                        "cpu_near2far_terms",
                        "cuda_near2far_transform_calls",
                        "cuda_near2far_terms",
                        "cuda_near2far_submitted_chunks",
                        "cuda_near2far_source_points",
                        "cuda_near2far_output_points",
                        "cuda_near2far_frequencies",
                        "cuda_near2far_periodic_copies",
                        "cuda_near2far_fast_precision_calls",
                        "cuda_near2far_mixed_precision_calls",
                        "cuda_near2far_cancellation_retries",
                        "cuda_near2far_target_tiles",
                        "cuda_near2far_frequency_tiles",
                        "cuda_near2far_operation_tiles",
                        "cuda_near2far_maximum_workspace_bytes",
                        "cuda_near2far_descriptor_uploads",
                        "cuda_near2far_kernel_launches",
                        "cuda_near2far_result_device_to_host_bytes",
                        "cuda_near2far_condition_device_to_host_bytes",
                        "near2far_dft_device_to_host_bytes_avoided",
                        "near2far_mpi_allreduce_calls",
                        "near2far_mpi_allreduce_bytes",
                        "cpu_near2far_adjoint_calls",
                        "cpu_near2far_adjoint_terms",
                        "cuda_near2far_adjoint_calls",
                        "cuda_near2far_adjoint_terms",
                        "cuda_near2far_adjoint_submitted_chunks",
                        "cuda_near2far_adjoint_source_points",
                        "cuda_near2far_adjoint_far_points",
                        "cuda_near2far_adjoint_frequencies",
                        "cuda_near2far_adjoint_periodic_copies",
                        "cuda_near2far_adjoint_fast_precision_calls",
                        "cuda_near2far_adjoint_mixed_precision_calls",
                        "cuda_near2far_adjoint_cancellation_retries",
                        "cuda_near2far_adjoint_maximum_workspace_bytes",
                        "cuda_near2far_adjoint_descriptor_uploads",
                        "cuda_near2far_adjoint_kernel_launches",
                        "cuda_near2far_adjoint_host_to_device_bytes",
                        "cuda_near2far_adjoint_result_device_to_host_bytes",
                        "cuda_near2far_adjoint_condition_device_to_host_bytes",
                    )
                ),
                "multi_gpu": values(
                    (
                        "mpi_messages",
                        "mpi_scalars",
                        "cuda_aware_bytes",
                        "pinned_staging_bytes",
                        "pinned_device_to_host_bytes",
                        "pinned_host_to_device_bytes",
                    )
                ),
                "mpi_completion": values(
                    (
                        "mpi_waitsome_executions",
                        "mpi_waitall_executions",
                    )
                ),
                "boundary_eh_overlap": values(
                    (
                        "boundary_eh_overlap_checks",
                        "boundary_eh_overlap_eligible",
                        "boundary_eh_overlap_launched_h",
                        "boundary_eh_overlap_launched_e",
                        "boundary_eh_overlap_skipped_disabled",
                        "boundary_eh_overlap_skipped_unsupported_schedule",
                        "boundary_eh_overlap_skipped_no_remote",
                        "boundary_eh_overlap_skipped_cold_topology",
                        "boundary_eh_overlap_rejected",
                    )
                ),
                "halo_curl_overlap": values(
                    (
                        "halo_curl_overlap_checks",
                        "halo_curl_overlap_eligible",
                        "halo_curl_overlap_launches",
                        "halo_curl_overlap_skipped_disabled",
                        "halo_curl_overlap_skipped_unsupported_schedule",
                        "halo_curl_overlap_skipped_no_remote",
                        "halo_curl_overlap_skipped_cold_topology",
                        "halo_curl_overlap_rejected_feature",
                        "halo_curl_overlap_rejected_small",
                        "halo_curl_overlap_full_points",
                        "halo_curl_overlap_interior_points",
                        "halo_curl_overlap_shell_points",
                    )
                ),
                "tile_coalescing": values(
                    (
                        "tile_coalesced_curl_chunk_phases",
                        "tile_coalesced_curl_input_tiles",
                        "tile_coalesced_update_eh_chunk_phases",
                        "tile_coalesced_update_eh_input_tiles",
                    )
                ),
                "phase_batch_policy": values(
                    (
                        "phase_curl_automatic_checks",
                        "phase_curl_automatic_selected",
                        "phase_curl_automatic_rejected",
                        "phase_curl_forced_batches",
                        "phase_curl_batched_operations",
                        "phase_curl_unbatched_operations",
                        "phase_curl_replay_checks",
                        "phase_curl_replay_hits",
                        "phase_curl_replay_unready",
                        "phase_curl_replay_generation_misses",
                        "phase_curl_replay_mirror_misses",
                        "phase_update_eh_automatic_checks",
                        "phase_update_eh_automatic_selected",
                        "phase_update_eh_automatic_rejected",
                        "phase_update_eh_forced_batches",
                        "phase_update_eh_batched_operations",
                        "phase_update_eh_unbatched_operations",
                    )
                ),
            }

    gpu = _GpuController()

    from .geom import (
        Block,
        Cone,
        Cylinder,
        DrudeSusceptibility,
        Ellipsoid,
        FreqRange,
        GeometricObject,
        GyrotropicDrudeSusceptibility,
        GyrotropicLorentzianSusceptibility,
        GyrotropicSaturatedSusceptibility,
        Lattice,
        LorentzianSusceptibility,
        MaterialGrid,
        Matrix,
        Medium,
        Mesh,
        MultilevelAtom,
        NoisyDrudeSusceptibility,
        NoisyLorentzianSusceptibility,
        Prism,
        Sphere,
        Susceptibility,
        Transition,
        Vector3,
        Wedge,
        check_nonnegative,
        geometric_object_duplicates,
        geometric_objects_duplicates,
        geometric_objects_lattice_duplicates,
        cartesian_to_lattice,
        lattice_to_cartesian,
        lattice_to_reciprocal,
        reciprocal_to_lattice,
        cartesian_to_reciprocal,
        reciprocal_to_cartesian,
        find_root_deriv,
        get_rotation_matrix,
    )
    from .simulation import (
        Absorber,
        BinaryPartition,
        Ldos,
        EnergyRegion,
        FluxRegion,
        ForceRegion,
        Harminv,
        PadeDFT,
        Identity,
        Mirror,
        ModeRegion,
        Near2FarRegion,
        PML,
        Rotate2,
        Rotate4,
        Simulation,
        Symmetry,
        DftObj,
        DftFlux,
        DftForce,
        DftNear2Far,
        DftEnergy,
        DftFields,
        Volume,
        DiffractedPlanewave,
        after_sources,
        after_sources_and_time,
        after_time,
        at_beginning,
        at_end,
        at_every,
        at_time,
        before_time,
        combine_step_funcs,
        complexarray,
        dft_ldos,
        display_progress,
        during_sources,
        GDSII_layers,
        GDSII_prisms,
        GDSII_vol,
        get_center_and_size,
        get_eigenmode_freqs,
        get_electric_energy,
        get_energy_freqs,
        get_flux_freqs,
        get_fluxes,
        get_force_freqs,
        get_forces,
        get_group_masters,
        get_ldos_freqs,
        get_magnetic_energy,
        get_near2far_freqs,
        get_num_groups,
        get_total_energy,
        in_point,
        in_volume,
        interpolate,
        merge_subgroup_data,
        output_epsilon,
        output_mu,
        output_hpwr,
        output_dpwr,
        output_tot_pwr,
        output_bfield,
        output_bfield_x,
        output_bfield_y,
        output_bfield_z,
        output_bfield_r,
        output_bfield_p,
        output_dfield,
        output_dfield_x,
        output_dfield_y,
        output_dfield_z,
        output_dfield_r,
        output_dfield_p,
        output_efield,
        output_efield_x,
        output_efield_y,
        output_efield_z,
        output_efield_r,
        output_efield_p,
        output_hfield,
        output_hfield_x,
        output_hfield_y,
        output_hfield_z,
        output_hfield_r,
        output_hfield_p,
        output_png,
        output_poynting,
        output_poynting_x,
        output_poynting_y,
        output_poynting_z,
        output_poynting_r,
        output_poynting_p,
        output_sfield,
        output_sfield_x,
        output_sfield_y,
        output_sfield_z,
        output_sfield_r,
        output_sfield_p,
        py_v3_to_vec,
        quiet,
        scale_energy_fields,
        scale_flux_fields,
        scale_force_fields,
        scale_near2far_fields,
        stop_after_walltime,
        stop_on_interrupt,
        stop_when_dft_decayed,
        stop_when_fields_decayed,
        stop_when_energy_decayed,
        stop_when_flux_decayed,
        synchronized_magnetic,
        to_appended,
        vec,
        verbosity,
        when_true,
        when_false,
        with_prefix
    )
    from .source import (
        ContinuousSource,
        CustomSource,
        EigenModeSource,
        GaussianSource,
        IndexedSource,
        Source,
        SourceTime,
        check_positive,
        GaussianBeamSource,
        GaussianBeam3DSource,
        GaussianBeam2DSource,
        get_equiv_sources,
    )
    from .verbosity_mgr import (
        Verbosity
    )

    _gpmeep_lazy_visualization_exports = frozenset(
        ("visualization", "plot2D", "plot3D", "plot_fields", "Animate2D")
    )

    def __getattr__(name):
        if name in _gpmeep_lazy_visualization_exports:
            from importlib import import_module as _gpmeep_import_module
            module = _gpmeep_import_module(".visualization", __name__)
            value = module if name == "visualization" else getattr(module, name)
            globals()[name] = value
            return value
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))

    def __dir__():
        return sorted(set(globals()) | _gpmeep_lazy_visualization_exports)

    if with_mpi():
        try:
            from mpi4py import MPI
        except ImportError as e:
            raise ImportError(
                "MPI-enabled Meep requires mpi4py to initialize and finalize "
                "its distributed runtime"
            ) from e
        else:
            # this variable reference is needed for lazy initialization of MPI
            comm = MPI.COMM_WORLD
            # mpi4py, rather than meep::initialize, owns MPI initialization in
            # Python.  Create the world-wide CUDA assignment table now, before
            # any Meep subcommunicators can be formed, and release it before
            # mpi4py finalizes MPI at process exit.
            _gpu_initialize_distributed_runtime()
            import atexit as _gpmeep_atexit
            _gpmeep_atexit.register(_gpu_finalize_distributed_runtime)
            if am_master():
                Procs=comm.Get_size()
                (Major,Minor)=MPI.Get_version();
                print('Using MPI version {}.{}, {} processes'.format(Major, Minor, Procs));

            if not am_master():
                import os
                import sys
                saved_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')

    vacuum = Medium(epsilon=1)
    air = Medium(epsilon=1)
    metal = Medium(epsilon=-inf)
    perfect_electric_conductor = Medium(epsilon=-inf)
    perfect_magnetic_conductor = Medium(mu=-inf)
    _t_start = wall_time()

    def report_elapsed_time():
        print("\nElapsed run time = {:.4f} s".format(wall_time() - _t_start))

    import atexit
    atexit.register(report_elapsed_time)
%}

%newobject create_structure;
%newobject _set_materials;
%inline %{

size_t get_realnum_size() {
  return sizeof(meep::realnum);
}

bool is_single_precision() {
  return sizeof(meep::realnum) == sizeof(float);
}

meep::structure *create_structure(vector3 cell_size,
                                    std::vector<meep_geom::dft_data> dft_data_list_,
                                    std::vector<meep::volume> pml_1d_vols_,
                                    std::vector<meep::volume> pml_2d_vols_,
                                    std::vector<meep::volume> pml_3d_vols_,
                                    std::vector<meep::volume> absorber_vols_,
                                    meep::grid_volume &gv,
                                    const meep::boundary_region &br,
                                    const meep::symmetry &sym,
                                    int num_chunks,
                                    double Courant,
                                    bool use_anisotropic_averaging,
                                    double tol,
                                    int maxeval,
                                    geometric_object_list gobj_list,
                                    vector3 center,
                                    bool _ensure_periodicity,
                                    meep_geom::material_type _default_material,
                                    meep_geom::absorber_list alist,
                                    meep_geom::material_type_list extra_materials,
                                    bool split_chunks_evenly,
                                    bool set_materials,
                                    meep::structure *existing_s,
                                    bool output_chunk_costs,
                                    const meep::binary_partition *my_bp) {
    // Initialize fragment_stats static members (used for creating chunks in choose_chunkdivision)
    meep_geom::fragment_stats::geom = gobj_list;
    meep_geom::fragment_stats::dft_data_list = dft_data_list_;
    meep_geom::fragment_stats::pml_1d_vols = pml_1d_vols_;
    meep_geom::fragment_stats::pml_2d_vols = pml_2d_vols_;
    meep_geom::fragment_stats::pml_3d_vols = pml_3d_vols_;
    meep_geom::fragment_stats::absorber_vols = absorber_vols_;
    meep_geom::fragment_stats::tol = tol;
    meep_geom::fragment_stats::maxeval = maxeval;
    meep_geom::fragment_stats::resolution = gv.a;
    meep_geom::fragment_stats::dims = gv.dim;
    meep_geom::fragment_stats::split_chunks_evenly = split_chunks_evenly;
    meep_geom::init_libctl(_default_material, _ensure_periodicity,
                           &gv, cell_size, center, &gobj_list);

    if (output_chunk_costs) {
         meep::volume thev = gv.surroundings();
         std::unique_ptr<meep::binary_partition> bp;
         if (!my_bp) bp = meep::choose_chunkdivision(gv, thev, num_chunks, sym);
         std::vector<grid_volume> chunk_vols;
         std::vector<int> ids;
         meep::split_by_binarytree(gv, chunk_vols, ids, (!my_bp) ? bp.get() : my_bp);
         for (size_t i = 0; i < chunk_vols.size(); ++i)
              master_printf("CHUNK:, %2zu, %f, %zu\n",i,chunk_vols[i].get_cost(),chunk_vols[i].surface_area());
         return NULL;
    }

    meep::structure *s;
    if (existing_s) {
      s = existing_s;
    }
    else {
      s = new meep::structure(gv, NULL, br, sym, num_chunks, Courant,
                              use_anisotropic_averaging, tol, maxeval, my_bp);
    }
    s->shared_chunks = true;

    return s;
}
meep_geom::geom_epsilon* _set_materials(meep::structure * s,
                    vector3 cell_size,
                    meep::grid_volume &gv,
                    bool use_anisotropic_averaging,
                    double tol,
                    int maxeval,
                    geometric_object_list gobj_list,
                    vector3 center,
                    bool _ensure_periodicity,
                    meep_geom::material_type _default_material,
                    meep_geom::absorber_list alist,
                    meep_geom::material_type_list extra_materials,
                    bool split_chunks_evenly,
                    bool set_materials,
                    meep_geom::geom_epsilon *existing_geps,
                    bool output_chunk_costs,
                    const meep::binary_partition *my_bp) {

    meep_geom::geom_epsilon *geps;
    if (existing_geps) {
        geps = existing_geps;
    } else {
        geps = meep_geom::make_geom_epsilon(s, &gobj_list, center, _ensure_periodicity, _default_material,
                                                extra_materials);
    }
    if (set_materials) {
        meep_geom::set_materials_from_geom_epsilon(s, geps, use_anisotropic_averaging, tol,
                                             maxeval,alist);
    }

    if (meep::verbosity > 1 && !split_chunks_evenly && set_materials) {
      int num_procs = meep::count_processors();
      double *costs = new double[num_procs];
      for (int i = 0; i < num_procs; i++)
        costs[i] = 0;
      for (int i = 0; i < s->num_chunks; i++)
        costs[s->chunks[i]->n_proc()] += s->chunks[i]->gv.get_cost();
      double sum = 0, sumsq = 0;
      master_printf("estimated costs per process: ");
      for (int i = 0; i < num_procs; i++) {
        double cost = costs[i];
        sum += cost;
        sumsq += cost*cost;
        master_printf("%g%s", cost, i == num_procs - 1 ? "\n" : ", ");
      }
      delete[] costs;
      double mean = sum / num_procs;
      double stddev = sumsq - num_procs * mean * mean;
      stddev = num_procs == 1 || stddev <= 0 ? 0.0 : sqrt(stddev / (num_procs - 1));
      master_printf("estimated cost mean = %g, stddev = %g\n", mean, stddev);
    }

    // Return params to default state
    meep_geom::fragment_stats::resolution = 0;
    meep_geom::fragment_stats::split_chunks_evenly = false;

    return geps;
}

void _get_epsilon_grid(geometric_object_list gobj_list,
                       meep_geom::material_type_list mlist,
                       meep_geom::material_type _default_material,
                       bool _ensure_periodicity,
                       meep::grid_volume gv,
                       vector3 cell_size,
                       vector3 cell_center,
                       int nx, double *xtics,
                       int ny, double *ytics,
                       int nz, double *ztics,
                       std::complex<double> *grid_vals,
                       double frequency) {
     meep_geom::get_epsilon_grid(gobj_list,
                                 mlist,
                                 _default_material,
                                 _ensure_periodicity,
                                 gv,
                                 cell_size,
                                 cell_center,
                                 nx, xtics,
                                 ny, ytics,
                                 nz, ztics,
                                 grid_vals,
                                 frequency);
}

%}

%pythoncode %{
# Preserve the historical wildcard surface.  Ordinary ``import meep`` keeps
# visualization lazy, while ``from meep import *`` explicitly resolves it.
__all__ = sorted(
    {name for name in globals() if not name.startswith("_")}
    | _gpmeep_lazy_visualization_exports
)
%}
