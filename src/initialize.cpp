/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software Foundation,
%  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>
#include <exception>

#include "meep.hpp"
#include "meep_internals.hpp"
#include "gpu_backend_internal.hpp"
#include "config.h"

// Cylindrical coordinates:

#ifdef HAVE_LIBGSL
#include <gsl/gsl_sf_bessel.h>
#endif

using namespace std;

namespace meep {

namespace {

class distributed_initialization_failure_guard {
public:
  explicit distributed_initialization_failure_guard(bool armed)
      : armed_(armed), completed_(false) {}
  void dismiss() noexcept { completed_ = true; }
  ~distributed_initialization_failure_guard() {
    // Do not infer failure from the process-wide C++11 uncaught-exception
    // flag: initialize_field may legally be called during an outer unwind.
    if (armed_ && !completed_)
      meep::abort(
          "rank-local failure during distributed field "
          "initialization; aborting the communicator to prevent an MPI "
          "collective or boundary deadlock");
  }

private:
  bool armed_;
  bool completed_;
};

} // namespace

#define J BesselJ
double J(int m, double kr) {
#if defined(HAVE_JN)
  return jn(m, kr); // POSIX/BSD jn function
#elif defined(HAVE_LIBGSL)
  return gsl_sf_bessel_Jn(m, kr);
#else
  meep::abort("not compiled with GSL, required for Bessel functions");
  return 0;
#endif
}
static double Jprime(int m, double kr) {
  if (m)
    return 0.5 * (J(m - 1, kr) - J(m + 1, kr));
  else
    return -J(1, kr);
}
static double Jroot(int m, int n) {
#ifdef HAVE_LIBGSL
  return gsl_sf_bessel_zero_Jnu(m, n + 1);
#else
  (void)m;
  (void)n;
  meep::abort("not compiled with GSL, required for Bessel functions");
  return 0;
#endif
}
static double Jmax(int m, int n) {
  double rlow, rhigh = Jroot(m, n), rtry;
  if (n == 0)
    rlow = 0;
  else
    rlow = Jroot(m, n - 1);
  double jplow = Jprime(m, rlow), jptry;
  do {
    rtry = rlow + (rhigh - rlow) * 0.5;
    jptry = Jprime(m, rtry);
    if (jplow * jptry < 0)
      rhigh = rtry;
    else
      rlow = rtry;
  } while (rhigh - rlow > rhigh * 1e-15);
  return rtry;
}

static double ktrans, kax;
static int m_for_J;
static complex<double> JJ(const vec &pt) {
  return polar(J(m_for_J, ktrans * pt.r()), kax * pt.r());
}
static complex<double> JP(const vec &pt) {
  return polar(Jprime(m_for_J, ktrans * pt.r()), kax * pt.r());
}

void fields::initialize_with_nth_te(int np0) {
  require_component(Hz);
  for (int i = 0; i < num_chunks; i++)
    chunks[i]->initialize_with_nth_te(np0, real(k[Z]));
}

void fields_chunk::initialize_with_nth_te(int np0, double kz) {
  const int im = int(m);
  const int n = (im == 0) ? np0 - 0 : np0 - 1;
  const double rmax = Jmax(im, n);
  ktrans = rmax * a / gv.nr();
  kax = kz * 2 * pi / a;
  m_for_J = im;
  initialize_field(Hz, JJ);
}

void fields::initialize_with_nth_tm(int np0) {
  require_component(Ez);
  require_component(Hp);
  for (int i = 0; i < num_chunks; i++)
    chunks[i]->initialize_with_nth_tm(np0, real(k[Z]));
}

void fields_chunk::initialize_with_nth_tm(int np1, double kz) {
  const int im = int(m);
  const int n = np1 - 1;
  const double rroot = Jroot(im, n);
  ktrans = rroot * a / gv.nr();
  kax = kz * 2 * pi / a;
  m_for_J = im;
  initialize_field(Ez, JJ);
  initialize_field(Hp, JP);
}

void fields::initialize_with_n_te(int ntot) {
  for (int n = 0; n < ntot; n++)
    initialize_with_nth_te(n + 1);
}

void fields::initialize_with_n_tm(int ntot) {
  for (int n = 0; n < ntot; n++)
    initialize_with_nth_tm(n + 1);
}

void fields::initialize_field(component c, complex<double> func(const vec &)) {
  // The user initializer itself may fail on one rank before peers enter a
  // CPU or CUDA boundary exchange, so keep this backend-independent.
  distributed_initialization_failure_guard distributed_failure_guard(
      count_processors() > 1);
  require_component(c);
  // initialize_field can execute boundary and E/H phases before the first
  // fields::step. Resolve the per-owner automatic workload policy first so a
  // launch-latency-dominated initialization cannot touch CUDA and only fall
  // back to the CPU on the subsequent time step.
  cuda_backend_preflight();
  cuda_step_preflight();
  for (int i = 0; i < num_chunks; i++)
    chunks[i]->initialize_field(c, func);
  step_boundaries(type(c));
  if (is_D(c)) {
    update_eh(E_stuff);
    step_boundaries(E_stuff);
  }
  if (is_B(c)) {
    update_eh(H_stuff);
    step_boundaries(H_stuff);
  }
  distributed_failure_guard.dismiss();
}

void fields_chunk::initialize_field(component c, complex<double> func(const vec &)) {
  if (f[c][0]) {
    // initialize_field mutates host arrays directly. Preserve any
    // device-authoritative state and invalidate mirrors before the callback
    // starts adding values.
    gpu::detail::sync_resident_cache_for_owner(this);
    gpu::detail::destroy_resident_cache_for_owner(this);
    // note: this loop is not thread-safe unless func is, which
    // isn't true if func e.g. calls back to Python
    LOOP_OVER_VOL(gv, c, i) {
      IVEC_LOOP_LOC(gv, here);
      complex<double> val = func(here);
      f[c][0][i] += real(val);
      if (!is_real) f[c][1][i] += imag(val);
    }
  }
}

} // namespace meep
