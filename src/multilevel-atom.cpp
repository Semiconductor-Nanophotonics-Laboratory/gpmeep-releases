/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

/* this file implements multilevel atomic materials for Meep */

#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include "meep.hpp"
#include "meep_internals.hpp"
#include "gpu_backend_internal.hpp"
#include "gpu_grid_index.hpp"
#include "config.h"

namespace meep {

multilevel_susceptibility::multilevel_susceptibility(int theL, int theT, const realnum *theGamma,
                                                     const realnum *theN0, const realnum *thealpha,
                                                     const realnum *theomega,
                                                     const realnum *thegamma,
                                                     const realnum *thesigmat) {
  L = theL;
  T = theT;
  Gamma = new realnum[L * L];
  memcpy(Gamma, theGamma, sizeof(realnum) * L * L);
  N0 = new realnum[L];
  memcpy(N0, theN0, sizeof(realnum) * L);
  alpha = new realnum[L * T];
  memcpy(alpha, thealpha, sizeof(realnum) * L * T);
  omega = new realnum[T];
  memcpy(omega, theomega, sizeof(realnum) * T);
  gamma = new realnum[T];
  memcpy(gamma, thegamma, sizeof(realnum) * T);
  sigmat = new realnum[T * 5];
  memcpy(sigmat, thesigmat, sizeof(realnum) * T * 5);
}

multilevel_susceptibility::multilevel_susceptibility(const multilevel_susceptibility &from)
    : susceptibility(from) {
  L = from.L;
  T = from.T;
  Gamma = new realnum[L * L];
  memcpy(Gamma, from.Gamma, sizeof(realnum) * L * L);
  N0 = new realnum[L];
  memcpy(N0, from.N0, sizeof(realnum) * L);
  alpha = new realnum[L * T];
  memcpy(alpha, from.alpha, sizeof(realnum) * L * T);
  omega = new realnum[T];
  memcpy(omega, from.omega, sizeof(realnum) * T);
  gamma = new realnum[T];
  memcpy(gamma, from.gamma, sizeof(realnum) * T);
  sigmat = new realnum[T * 5];
  memcpy(sigmat, from.sigmat, sizeof(realnum) * T * 5);
}

multilevel_susceptibility::~multilevel_susceptibility() {
  delete[] Gamma;
  delete[] N0;
  delete[] alpha;
  delete[] omega;
  delete[] gamma;
  delete[] sigmat;
}

#if MEEP_SINGLE
#define DGETRF F77_FUNC(sgetrf, SGETRF)
#define DGETRI F77_FUNC(sgetri, SGETRI)
#else
#define DGETRF F77_FUNC(dgetrf, DGETRF)
#define DGETRI F77_FUNC(dgetri, DGETRI)
#endif
extern "C" void DGETRF(const int *m, const int *n, realnum *A, const int *lda, int *ipiv,
                       int *info);
extern "C" void DGETRI(const int *n, realnum *A, const int *lda, int *ipiv, realnum *work,
                       int *lwork, int *info);

/* S -> inv(S), where S is a p x p matrix in row-major order */
static bool invert(realnum *S, int p) {
#ifdef HAVE_LAPACK
  int info = 0;
  int *ipiv = new int[p];
  DGETRF(&p, &p, S, &p, ipiv, &info);
  if (info < 0) meep::abort("invalid argument %d in DGETRF", -info);
  if (info > 0) {
    delete[] ipiv;
    return false;
  } // singular

  int lwork = -1;
  realnum work1 = 0.0;
  DGETRI(&p, S, &p, ipiv, &work1, &lwork, &info);
  if (info != 0) meep::abort("error %d in DGETRI workspace query", info);
  lwork = int(work1);
  realnum *work = new realnum[lwork]();
  DGETRI(&p, S, &p, ipiv, work, &lwork, &info);
  if (info < 0) meep::abort("invalid argument %d in DGETRI", -info);

  delete[] work;
  delete[] ipiv;
  return info == 0;
#else /* !HAVE_LAPACK */
  meep::abort("LAPACK is needed for multilevel-atom support");
  return false;
#endif
}

typedef realnum *realnumP;
typedef struct {
  size_t sz_data;
  size_t ntot;
  realnum *GammaInv;                    // inv(1 + Gamma * dt / 2)
  realnumP *P[NUM_FIELD_COMPONENTS][2]; // P[c][cmp][transition][i]
  realnumP *P_prev[NUM_FIELD_COMPONENTS][2];
  realnum *N;    // ntot x L array of centered grid populations N[i*L + level]
  realnum *Ntmp; // ntot x L population-update scratch
  realnum data[1];
} multilevel_data;

static size_t checked_multilevel_product(size_t left, size_t right,
                                         const char *what) {
  if (left && right > std::numeric_limits<size_t>::max() / left)
    meep::abort("multilevel_susceptibility: %s size overflow", what);
  return left * right;
}

static size_t checked_multilevel_sum(size_t left, size_t right,
                                     const char *what) {
  if (right > std::numeric_limits<size_t>::max() - left)
    meep::abort("multilevel_susceptibility: %s size overflow", what);
  return left + right;
}

void *multilevel_susceptibility::new_internal_data(realnum *W[NUM_FIELD_COMPONENTS][2],
                                                   const grid_volume &gv) const {
  if (L <= 0 || T <= 0)
    meep::abort(
        "multilevel_susceptibility requires positive level and transition "
        "counts");
  const size_t ntot = gv.ntot();
  size_t num = 0; // number of P scalars per transition
  FOR_COMPONENTS(c) DOCMP2 {
    if (needs_P(c, cmp, W))
      num = checked_multilevel_sum(
          num, checked_multilevel_product(2, ntot, "polarization"),
          "polarization");
  }
  const size_t matrix_count = checked_multilevel_product(
      static_cast<size_t>(L), static_cast<size_t>(L), "matrix");
  const size_t population_count = checked_multilevel_product(
      ntot, static_cast<size_t>(L), "population");
  const size_t polarization_count = checked_multilevel_product(
      num, static_cast<size_t>(T), "polarization");
  size_t scalar_count =
      checked_multilevel_sum(matrix_count, population_count, "state");
  scalar_count =
      checked_multilevel_sum(scalar_count, population_count, "scratch");
  scalar_count =
      checked_multilevel_sum(scalar_count, polarization_count, "state");
  const size_t payload_bytes =
      checked_multilevel_product(scalar_count, sizeof(realnum), "allocation");
  const size_t header_bytes = offsetof(multilevel_data, data);
  const size_t sz =
      checked_multilevel_sum(header_bytes, payload_bytes, "allocation");
  multilevel_data *d = (multilevel_data *)malloc(sz);
  if (d == NULL) meep::abort("%s:%i:out of memory(%lu)", __FILE__, __LINE__, sz);
  memset(d, 0, sz);
  d->sz_data = sz;
  return (void *)d;
}

void multilevel_susceptibility::init_internal_data(realnum *W[NUM_FIELD_COMPONENTS][2], realnum dt,
                                                   const grid_volume &gv, void *data) const {
  multilevel_data *d = (multilevel_data *)data;
  size_t sz_data = d->sz_data;
  memset(d, 0, sz_data);
  d->sz_data = sz_data;
  size_t ntot = d->ntot = gv.ntot();

  /* d->data points to a big block of data that holds GammaInv, P,
     P_prev, Ntmp, and N.  We also initialize a bunch of convenience
     pointer in d to point to the corresponding data in d->data, so
     that we don't have to remember in other functions how d->data is
     laid out. */

  d->GammaInv = d->data;
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      d->GammaInv[i * L + j] = (i == j) + Gamma[i * L + j] * dt / 2;
  if (!invert(d->GammaInv, L))
    meep::abort("multilevel_susceptibility: I + Gamma*dt/2 matrix singular");

  realnum *P = d->data + L * L;
  realnum *P_prev = P + ntot;
  FOR_COMPONENTS(c) DOCMP2 {
    if (needs_P(c, cmp, W)) {
      d->P[c][cmp] = new realnumP[T];
      d->P_prev[c][cmp] = new realnumP[T];
      for (int t = 0; t < T; ++t) {
        d->P[c][cmp][t] = P;
        d->P_prev[c][cmp][t] = P_prev;
        P += 2 * ntot;
        P_prev += 2 * ntot;
      }
    }
  }

  d->Ntmp = P;
  d->N = P + ntot * L; // the last L*ntot block of the data

  // initial populations
  for (size_t i = 0; i < ntot; ++i)
    for (int l = 0; l < L; ++l)
      d->N[i * L + l] = N0[l];
}

void multilevel_susceptibility::delete_internal_data(void *data) const {
  if (data) {
    multilevel_data *d = (multilevel_data *)data;
    FOR_COMPONENTS(c) DOCMP2 {
      delete[] d->P[c][cmp];
      delete[] d->P_prev[c][cmp];
    }
    free(data);
  }
}

void *multilevel_susceptibility::copy_internal_data(void *data) const {
  multilevel_data *d = (multilevel_data *)data;
  if (!d) return 0;
  multilevel_data *dnew = (multilevel_data *)malloc(d->sz_data);
  memcpy(dnew, d, d->sz_data);
  size_t ntot = d->ntot;
  dnew->GammaInv = dnew->data;
  realnum *P = dnew->data + L * L;
  realnum *P_prev = P + ntot;
  FOR_COMPONENTS(c) DOCMP2 {
    if (d->P[c][cmp]) {
      dnew->P[c][cmp] = new realnumP[T];
      dnew->P_prev[c][cmp] = new realnumP[T];
      for (int t = 0; t < T; ++t) {
        dnew->P[c][cmp][t] = P;
        dnew->P_prev[c][cmp][t] = P_prev;
        P += 2 * ntot;
        P_prev += 2 * ntot;
      }
    }
  }
  dnew->Ntmp = P;
  dnew->N = P + ntot * L;
  return (void *)dnew;
}

int multilevel_susceptibility::num_cinternal_notowned_needed(component c,
                                                             void *P_internal_data) const {
  multilevel_data *d = (multilevel_data *)P_internal_data;
  return d->P[c][0] ? T : 0;
}

realnum *multilevel_susceptibility::cinternal_notowned_ptr(int inotowned, component c, int cmp,
                                                           int n, void *P_internal_data) const {
  multilevel_data *d = (multilevel_data *)P_internal_data;
  if (!d || !d->P[c][cmp] || inotowned < 0 || inotowned >= T) // never true
    return NULL;
  return d->P[c][cmp][inotowned] + n;
}

void multilevel_susceptibility::update_P(realnum *W[NUM_FIELD_COMPONENTS][2],
                                         realnum *W_prev[NUM_FIELD_COMPONENTS][2], realnum dt,
                                         const grid_volume &gv, void *P_internal_data) const {
  multilevel_data *d = (multilevel_data *)P_internal_data;
  realnum dt2 = 0.5 * dt;

  // field directions and offsets for E * dP dot product.
  component cdot[3] = {Dielectric, Dielectric, Dielectric};
  ptrdiff_t o1[3], o2[3];
  int idot = 0;
  FOR_COMPONENTS(c) {
    if (d->P[c][0]) {
      if (idot == 3) meep::abort("bug in meep: too many polarization components");
      gv.yee2cent_offsets(c, o1[idot], o2[idot]);
      cdot[idot++] = c;
    }
  }

  // update N from W and P
  realnum *GammaInv = d->GammaInv;
  LOOP_OVER_VOL_OWNED(gv, Centered, i) {
    realnum *N = d->N + i * L; // N at current point, to update
    realnum *Ntmp = d->Ntmp + i * L;

    // Ntmp = (I - Gamma * dt/2) * N
    for (int l1 = 0; l1 < L; ++l1) {
      Ntmp[l1] = 0;
      for (int l2 = 0; l2 < L; ++l2) {
        Ntmp[l1] += ((l1 == l2) - Gamma[l1 * L + l2] * dt2) * N[l2];
      }
    }

    // compute E*8 at point i
    realnum E8[3][2] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    for (idot = 0; idot < 3 && cdot[idot] != Dielectric; ++idot) {
      realnum *w = W[cdot[idot]][0], *wp = W_prev[cdot[idot]][0];
      E8[idot][0] = w[i] + w[i + o1[idot]] + w[i + o2[idot]] + w[i + o1[idot] + o2[idot]] + wp[i] +
                    wp[i + o1[idot]] + wp[i + o2[idot]] + wp[i + o1[idot] + o2[idot]];
      if (W[cdot[idot]][1]) {
        w = W[cdot[idot]][1];
        wp = W_prev[cdot[idot]][1];
        E8[idot][1] = w[i] + w[i + o1[idot]] + w[i + o2[idot]] + w[i + o1[idot] + o2[idot]] +
                      wp[i] + wp[i + o1[idot]] + wp[i + o2[idot]] + wp[i + o1[idot] + o2[idot]];
      }
      else
        E8[idot][1] = 0;
    }

    // Ntmp = Ntmp + alpha * E * dP
    for (int t = 0; t < T; ++t) {
      // compute 32 * E * dP and 64 * E * P at point i
      realnum EdP32 = 0;
      realnum EPave64 = 0;
      realnum gperpdt = gamma[t] * pi * dt;
      for (idot = 0; idot < 3 && cdot[idot] != Dielectric; ++idot) {
        realnum *p = d->P[cdot[idot]][0][t], *pp = d->P_prev[cdot[idot]][0][t];
        realnum dP = p[i] + p[i + o1[idot]] + p[i + o2[idot]] + p[i + o1[idot] + o2[idot]] -
                     (pp[i] + pp[i + o1[idot]] + pp[i + o2[idot]] + pp[i + o1[idot] + o2[idot]]);
        realnum Pave2 = p[i] + p[i + o1[idot]] + p[i + o2[idot]] + p[i + o1[idot] + o2[idot]] +
                        (pp[i] + pp[i + o1[idot]] + pp[i + o2[idot]] + pp[i + o1[idot] + o2[idot]]);
        EdP32 += dP * E8[idot][0];
        EPave64 += Pave2 * E8[idot][0];
        if (d->P[cdot[idot]][1]) {
          p = d->P[cdot[idot]][1][t];
          pp = d->P_prev[cdot[idot]][1][t];
          dP = p[i] + p[i + o1[idot]] + p[i + o2[idot]] + p[i + o1[idot] + o2[idot]] -
               (pp[i] + pp[i + o1[idot]] + pp[i + o2[idot]] + pp[i + o1[idot] + o2[idot]]);
          Pave2 = p[i] + p[i + o1[idot]] + p[i + o2[idot]] + p[i + o1[idot] + o2[idot]] +
                  (pp[i] + pp[i + o1[idot]] + pp[i + o2[idot]] + pp[i + o1[idot] + o2[idot]]);
          EdP32 += dP * E8[idot][1];
          EPave64 += Pave2 * E8[idot][1];
        }
      }
      EdP32 *= 0.03125;    /* divide by 32 */
      EPave64 *= 0.015625; /* divide by 64 (extra factor of 1/2 is from P_current + P_previous) */
      for (int l = 0; l < L; ++l)
        Ntmp[l] += alpha[l * T + t] * EdP32 + alpha[l * T + t] * gperpdt * EPave64;
    }

    // N = GammaInv * Ntmp
    for (int l1 = 0; l1 < L; ++l1) {
      N[l1] = 0;
      for (int l2 = 0; l2 < L; ++l2)
        N[l1] += GammaInv[l1 * L + l2] * Ntmp[l2];
    }
  }

  // each P is updated as a damped harmonic oscillator
  for (int t = 0; t < T; ++t) {
    const realnum omega2pi = 2 * pi * omega[t], g2pi = gamma[t] * 2 * pi, gperp = gamma[t] * pi;
    const realnum omega0dtsqrCorrected = omega2pi * omega2pi * dt * dt + gperp * gperp * dt * dt;
    const realnum gamma1inv = 1 / (1 + g2pi * dt2), gamma1 = (1 - g2pi * dt2);
    const realnum dtsqr = dt * dt;
    // note that gamma[t]*2*pi = 2*gamma_perp as one would usually write it in SALT. -- AWC

    // figure out which levels this transition couples
    int lp = -1, lm = -1;
    for (int l = 0; l < L; ++l) {
      if (alpha[l * T + t] > 0) lp = l;
      if (alpha[l * T + t] < 0) lm = l;
    }
    if (lp < 0 || lm < 0) meep::abort("invalid alpha array for transition %d", t);

    FOR_COMPONENTS(c) DOCMP2 {
      if (d->P[c][cmp]) {
        const realnum *w = W[c][cmp], *s = sigma[c][component_direction(c)];
        const realnum st = sigmat[5 * t + component_direction(c)];
        if (w && s) {
          realnum *p = d->P[c][cmp][t], *pp = d->P_prev[c][cmp][t];

          ptrdiff_t o1, o2;
          gv.cent2yee_offsets(c, o1, o2);
          o1 *= L;
          o2 *= L;
          const realnum *N = d->N;

          // directions/strides for offdiagonal terms, similar to update_eh
          const direction d = component_direction(c);
          direction d1 = cycle_direction(gv.dim, d, 1);
          component c1 = direction_component(c, d1);
          const realnum *w1 = W[c1][cmp];
          const realnum *s1 = w1 ? sigma[c][d1] : NULL;
          direction d2 = cycle_direction(gv.dim, d, 2);
          component c2 = direction_component(c, d2);
          const realnum *w2 = W[c2][cmp];
          const realnum *s2 = w2 ? sigma[c][d2] : NULL;

          if (s1 || s2) { meep::abort("nondiagonal saturable gain is not yet supported"); }
          else { // isotropic
            LOOP_OVER_VOL_OWNED(gv, c, i) {
              realnum pcur = p[i];
              const realnum *Ni = N + i * L;
              // dNi is population inversion for this transition
              realnum dNi = 0.25 * (Ni[lp] + Ni[lp + o1] + Ni[lp + o2] + Ni[lp + o1 + o2] - Ni[lm] -
                                    Ni[lm + o1] - Ni[lm + o2] - Ni[lm + o1 + o2]);
              p[i] = gamma1inv * (pcur * (2 - omega0dtsqrCorrected) - gamma1 * pp[i] -
                                  dtsqr * (st * s[i] * w[i]) * dNi);
              pp[i] = pcur;
            }
          }
        }
      }
    }
  }
}

bool multilevel_susceptibility::cuda_update_P_eligible(
    realnum *W[NUM_FIELD_COMPONENTS][2],
    realnum *W_prev[NUM_FIELD_COMPONENTS][2], const grid_volume &gv,
    std::string *reason) const {
  (void)W_prev;
  const auto reject = [reason](const char *message) {
    if (reason) *reason = message;
    return false;
  };
  if (sizeof(realnum) != sizeof(float))
    return reject("multilevel CUDA stepping requires FP32 fields");
  if (L <= 0 || T <= 0 || !Gamma || !N0 || !alpha || !omega ||
      !gamma || !sigmat)
    return reject("multilevel CUDA stepping requires valid material arrays");

  for (int t = 0; t < T; ++t) {
    int upper = -1;
    int lower = -1;
    for (int l = 0; l < L; ++l) {
      if (alpha[l * T + t] > 0) upper = l;
      if (alpha[l * T + t] < 0) lower = l;
    }
    if (upper < 0 || lower < 0)
      return reject(
          "multilevel CUDA stepping requires one positive and one negative "
          "level coefficient per transition");
  }

  int physical_components = 0;
  FOR_COMPONENTS(c) {
    if (!needs_P(c, 0, W)) continue;
    if (++physical_components > 3)
      return reject(
          "multilevel CUDA stepping supports at most three field "
          "components");
  }
  FOR_COMPONENTS(c) DOCMP2 {
    if (!needs_P(c, cmp, W)) continue;
    const direction primary = component_direction(c);
    // f_w_prev is allocated lazily by update_eh after the whole-step
    // preflight and before update_pols. Its layout is validated again in
    // update_P_cuda before either multilevel kernel is launched.
    if (!W[c][cmp])
      return reject(
          "multilevel CUDA stepping requires primary fields");
    const direction d1 = cycle_direction(gv.dim, primary, 1);
    const component c1 = direction_component(c, d1);
    const direction d2 = cycle_direction(gv.dim, primary, 2);
    const component c2 = direction_component(c, d2);
    if ((W[c1][cmp] && sigma[c][d1]) ||
        (W[c2][cmp] && sigma[c][d2]))
      return reject(
          "multilevel CUDA stepping does not support nondiagonal "
          "saturable gain");
  }
  if (reason) reason->clear();
  return true;
}

bool multilevel_susceptibility::update_P_cuda(
    gpu::detail::resident_cache *cache,
    realnum *W[NUM_FIELD_COMPONENTS][2],
    realnum *W_prev[NUM_FIELD_COMPONENTS][2], realnum dt,
    const grid_volume &gv, void *P_internal_data) const {
#if MEEP_HAVE_CUDA
  if (!cache || !P_internal_data || sizeof(realnum) != sizeof(float))
    return false;
  std::string eligibility_reason;
  if (!cuda_update_P_eligible(W, W_prev, gv, &eligibility_reason))
    throw std::runtime_error(eligibility_reason);

  multilevel_data *d =
      static_cast<multilevel_data *>(P_internal_data);
  if (d->ntot != gv.ntot() || !d->GammaInv || !d->Ntmp || !d->N)
    throw std::runtime_error(
        "CUDA multilevel internal state has an invalid layout");

  std::vector<gpu::detail::multilevel_population_channel_fp32>
      population_channels;
  std::vector<gpu::detail::multilevel_polarization_channel_fp32>
      polarization_channels;
  std::vector<gpu::detail::multilevel_transition_fp32> transitions(
      static_cast<size_t>(T));

  // Match the CPU accumulation order: component, then real/imaginary.
  FOR_COMPONENTS(c) {
    if (!d->P[c][0]) continue;
    ptrdiff_t centered_offset1 = 0;
    ptrdiff_t centered_offset2 = 0;
    gv.yee2cent_offsets(c, centered_offset1, centered_offset2);
    for (int cmp = 0; cmp < 2; ++cmp) {
      if (!d->P[c][cmp]) continue;
      if (!d->P[c][cmp][0] || !d->P_prev[c][cmp][0] ||
          !W[c][cmp] || !W_prev[c][cmp])
        throw std::runtime_error(
            "CUDA multilevel population channel has an invalid layout");
      for (int t = 0; t < T; ++t) {
        const ptrdiff_t expected =
            static_cast<ptrdiff_t>(2) * t *
            static_cast<ptrdiff_t>(d->ntot);
        if (d->P[c][cmp][t] != d->P[c][cmp][0] + expected ||
            d->P_prev[c][cmp][t] !=
                d->P[c][cmp][t] +
                    static_cast<ptrdiff_t>(d->ntot))
          throw std::runtime_error(
              "CUDA multilevel transition arrays are not contiguous");
      }
      population_channels.push_back(
          {W[c][cmp], W_prev[c][cmp], d->P[c][cmp][0],
           centered_offset1, centered_offset2});
    }
  }

  FOR_COMPONENTS(c) DOCMP2 {
    if (!d->P[c][cmp]) continue;
    const direction primary = component_direction(c);
    const realnum *field = W[c][cmp];
    const realnum *diagonal_sigma = sigma[c][primary];
    if (!field || !diagonal_sigma) continue;
    const direction d1 = cycle_direction(gv.dim, primary, 1);
    const component c1 = direction_component(c, d1);
    const direction d2 = cycle_direction(gv.dim, primary, 2);
    const component c2 = direction_component(c, d2);
    if ((W[c1][cmp] && sigma[c][d1]) ||
        (W[c2][cmp] && sigma[c][d2]))
      throw std::runtime_error(
          "CUDA multilevel media do not support nondiagonal saturable "
          "gain");
    ptrdiff_t population_offset1 = 0;
    ptrdiff_t population_offset2 = 0;
    gv.cent2yee_offsets(c, population_offset1, population_offset2);
    const gpu::detail::index_space_fp32 index_space =
        gpu::detail::make_index_space_fp32(
            gv, gv.little_owned_corner(c), gv.big_corner());
    const size_t point_count =
        index_space.extent1 * index_space.extent2 *
        index_space.extent3;
    polarization_channels.push_back(
        {d->P[c][cmp][0], field, diagonal_sigma, index_space,
         point_count, population_offset1, population_offset2,
         static_cast<int>(primary)});
  }

  const realnum dt_half = 0.5 * dt;
  const realnum dt_squared = dt * dt;
  for (int t = 0; t < T; ++t) {
    int upper = -1;
    int lower = -1;
    for (int l = 0; l < L; ++l) {
      if (alpha[l * T + t] > 0) upper = l;
      if (alpha[l * T + t] < 0) lower = l;
    }
    if (upper < 0 || lower < 0)
      throw std::runtime_error(
          "CUDA multilevel transition has an invalid alpha column");
    const realnum omega_2pi = 2 * pi * omega[t];
    const realnum gamma_2pi = 2 * pi * gamma[t];
    const realnum gamma_perpendicular = pi * gamma[t];
    gpu::detail::multilevel_transition_fp32 &transition =
        transitions[static_cast<size_t>(t)];
    transition.upper_level = upper;
    transition.lower_level = lower;
    transition.population_damping =
        static_cast<float>(gamma_perpendicular * dt);
    transition.diagonal = static_cast<float>(
        2 - omega_2pi * omega_2pi * dt_squared -
        gamma_perpendicular * gamma_perpendicular * dt_squared);
    transition.gamma_previous =
        static_cast<float>(1 - gamma_2pi * dt_half);
    transition.gamma_inverse =
        static_cast<float>(1 / (1 + gamma_2pi * dt_half));
    for (int direction = 0; direction < 5; ++direction)
      transition.drive_scale[direction] =
          static_cast<float>(
              dt_squared * sigmat[5 * t + direction]);
  }

  const gpu::detail::index_space_fp32 centered_space =
      gpu::detail::make_index_space_fp32(
          gv, gv.little_owned_corner(Centered), gv.big_corner());
  gpu::detail::resident_update_multilevel_fp32(
      cache, d->N, d->Ntmp, Gamma, d->GammaInv, alpha,
      static_cast<size_t>(L), static_cast<size_t>(T), d->ntot,
      static_cast<float>(dt_half), centered_space,
      population_channels.data(), population_channels.size(),
      polarization_channels.data(), polarization_channels.size(),
      transitions.data());
  return true;
#else
  (void)cache;
  (void)W;
  (void)W_prev;
  (void)dt;
  (void)gv;
  (void)P_internal_data;
  return false;
#endif
}

bool multilevel_susceptibility::subtract_P_cuda(
    gpu::detail::resident_cache *cache, field_type ft,
    realnum *f_minus_p[NUM_FIELD_COMPONENTS][2],
    void *P_internal_data) const {
#if MEEP_HAVE_CUDA
  if (!cache || !P_internal_data || sizeof(realnum) != sizeof(float))
    return false;
  multilevel_data *d =
      static_cast<multilevel_data *>(P_internal_data);
  const field_type ft2 = ft == E_stuff ? D_stuff : B_stuff;
  for (int t = 0; t < T; ++t) {
    FOR_FT_COMPONENTS(ft, ec) DOCMP2 {
      if (!d->P[ec][cmp]) continue;
      const component dc = field_type_component(ft2, ec);
      if (f_minus_p[dc][cmp])
        gpu::detail::resident_subtract_fp32(
            cache, f_minus_p[dc][cmp], d->P[ec][cmp][t], d->ntot);
    }
  }
  return true;
#else
  (void)cache;
  (void)ft;
  (void)f_minus_p;
  (void)P_internal_data;
  return false;
#endif
}

bool multilevel_susceptibility::prepare_boundary_cuda(
    gpu::detail::resident_cache *cache,
    void *P_internal_data) const {
#if MEEP_HAVE_CUDA
  if (!cache || !P_internal_data || sizeof(realnum) != sizeof(float))
    return false;
  multilevel_data *d =
      static_cast<multilevel_data *>(P_internal_data);
  if (!d->ntot || T <= 0)
    throw std::runtime_error(
        "CUDA multilevel boundary state has an invalid layout");
  const size_t pair_count = checked_multilevel_product(
      static_cast<size_t>(2), static_cast<size_t>(T),
      "boundary polarization");
  const size_t scalar_count = checked_multilevel_product(
      pair_count, d->ntot, "boundary polarization");
  FOR_COMPONENTS(c) DOCMP2 {
    if (!d->P[c][cmp]) continue;
    if (!d->P[c][cmp][0])
      throw std::runtime_error(
          "CUDA multilevel boundary polarization is null");
    gpu::detail::resident_ensure_mirror_fp32(
        cache, d->P[c][cmp][0], scalar_count);
  }
  return true;
#else
  (void)cache;
  (void)P_internal_data;
  return false;
#endif
}

void multilevel_susceptibility::subtract_P(field_type ft,
                                           realnum *f_minus_p[NUM_FIELD_COMPONENTS][2],
                                           void *P_internal_data) const {
  multilevel_data *d = (multilevel_data *)P_internal_data;
  field_type ft2 = ft == E_stuff ? D_stuff : B_stuff; // for sources etc.
  size_t ntot = d->ntot;
  for (int t = 0; t < T; ++t) {
    FOR_FT_COMPONENTS(ft, ec) DOCMP2 {
      if (d->P[ec][cmp]) {
        component dc = field_type_component(ft2, ec);
        if (f_minus_p[dc][cmp]) {
          realnum *p = d->P[ec][cmp][t];
          realnum *fmp = f_minus_p[dc][cmp];
          for (size_t i = 0; i < ntot; ++i)
            fmp[i] -= p[i];
        }
      }
    }
  }
}

} // namespace meep
