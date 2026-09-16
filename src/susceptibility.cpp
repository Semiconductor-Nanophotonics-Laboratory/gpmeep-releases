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

/* This file implements dispersive materials for Meep via a
   polarization P = \chi(\omega) W, where W is e.g. E or H.  Each
   subclass of the susceptibility class should implement a different
   type of \chi(\omega).  The subclass knows how to timestep P given W
   at the current (and possibly previous) timestep, and any additional
   internal data that needs to be allocated along with P.

   Each \chi(\omega) is spatially multiplied by a (scalar) sigma
   array.  The meep::fields class is responsible for allocating P and
   sigma and passing them to susceptibility::update_P. */

#include <stdlib.h>
#include <string.h>
#include "meep.hpp"
#include "meep_internals.hpp"
#include "gpu_backend_internal.hpp"
#include "gpu_grid_index.hpp"
#include "meep_cuda/detail/polarization.hpp"

using namespace std;

namespace meep {

int susceptibility::cur_id = 0;

susceptibility *susceptibility::clone() const {
  susceptibility *sus = new susceptibility(*this);
  sus->next = 0;
  sus->ntot = ntot;
  sus->id = id;
  FOR_COMPONENTS(c) FOR_DIRECTIONS(d) {
    if (sigma[c][d]) {
      sus->sigma[c][d] = new realnum[ntot];
      memcpy(sus->sigma[c][d], sigma[c][d], sizeof(realnum) * ntot);
    }
    else
      sus->sigma[c][d] = NULL;
    sus->trivial_sigma[c][d] = trivial_sigma[c][d];
  }
  return sus;
}

// generic base class definition.
std::complex<realnum> susceptibility::chi1(realnum freq, realnum sigma) {
  (void)freq;
  (void)sigma;
  return std::complex<realnum>(0, 0);
}

void susceptibility::delete_internal_data(void *data) const { free(data); }

/* Return whether or not we need to allocate P[c][cmp].  (We don't need to
   allocate P[c] if we can be sure it will be zero.)

   We are a bit wasteful because if sigma is nontrivial in *any* chunk,
   we allocate the corresponding P on *every* owned chunk.  This greatly
   simplifies communication in boundaries.cpp, because we can be sure that
   one chunk has a P then any chunk it borders has the same P, so we don't
   have to worry about communicating with something that doesn't exist.
   TODO: reduce memory usage (bookkeeping seem much harder, though).
*/
bool susceptibility::needs_P(component c, int cmp, realnum *W[NUM_FIELD_COMPONENTS][2]) const {
  if (!is_electric(c) && !is_magnetic(c)) return false;
  FOR_DIRECTIONS(d) {
    if (!trivial_sigma[c][d] && W[direction_component(c, d)][cmp]) return true;
  }
  return false;
}

/* return whether we need the notowned parts of the W field --
   by default, this is only the case if sigma has offdiagonal components
   coupling P to W.   (See needs_P: again, this true if the notowned
   W is needed in *any* chunk.) */
bool susceptibility::needs_W_notowned(component c, realnum *W[NUM_FIELD_COMPONENTS][2]) const {
  FOR_DIRECTIONS(d) {
    if (d != component_direction(c)) {
      component cP = direction_component(c, d);
      if (needs_P(cP, 0, W) && !trivial_sigma[cP][component_direction(c)]) return true;
    }
  }
  return false;
}

typedef struct {
  size_t sz_data;
  size_t ntot;
  gpu::detail::lorentzian_state_kind state_kind;
  realnum *P[NUM_FIELD_COMPONENTS][2];
  realnum *P_auxiliary[NUM_FIELD_COMPONENTS][2];
  realnum data[1];
} lorentzian_data;

static gpu::detail::lorentzian_state_kind lorentzian_state_kind_for(
    bool no_omega_0_denominator, realnum omega_0, realnum gamma,
    realnum dt) {
  if (no_omega_0_denominator)
    return gpu::detail::lorentzian_state_kind::increment;
#if MEEP_SINGLE
  // When the pole advances by less than 0.1 radian per step, P and P_prev are
  // close and the previous-P recurrence repeatedly subtracts nearly equal
  // states while also forming 2-(omega*dt)^2.  The algebraically equivalent
  // increment state retains the restoring force for lossy poles.  Preserve
  // previous-P for very weak-loss poles because it resolves their tiny decay
  // rate better, unless forming 2-(omega*dt)^2 has itself lost more than
  // sqrt(epsilon) of the restoring coefficient.
  const realnum omega_dt = 2 * pi * omega_0 * dt;
  const realnum omega_dt_squared = omega_dt * omega_dt;
  const realnum represented_restoring_term =
      realnum(2) - (realnum(2) - omega_dt_squared);
  const realnum relative_restoring_term_error =
      omega_dt_squared == 0
          ? 0
          : std::abs(represented_restoring_term - omega_dt_squared) /
                omega_dt_squared;
  const realnum conditioning_threshold =
      std::sqrt(std::numeric_limits<realnum>::epsilon());
  const bool lossy_small_phase =
      omega_dt_squared < realnum(0.01) &&
      std::abs(gamma) > conditioning_threshold * std::abs(omega_0);
  if (relative_restoring_term_error > conditioning_threshold ||
      lossy_small_phase)
    return gpu::detail::lorentzian_state_kind::increment;
#else
  (void)omega_0;
  (void)dt;
#endif
  return gpu::detail::lorentzian_state_kind::previous;
}

static void require_lorentzian_state_kind(
    const lorentzian_data *data, bool no_omega_0_denominator,
    realnum omega_0, realnum gamma, realnum dt) {
  if (!data ||
      data->state_kind !=
          lorentzian_state_kind_for(no_omega_0_denominator, omega_0, gamma,
                                    dt))
    throw std::runtime_error(
        "Lorentzian/Drude auxiliary-state representation mismatch");
}

static void require_valid_lorentzian_state_kind(
    const lorentzian_data *data) {
  if (!data ||
      (data->state_kind != gpu::detail::lorentzian_state_kind::previous &&
       data->state_kind != gpu::detail::lorentzian_state_kind::increment))
    throw std::runtime_error(
        "Invalid Lorentzian/Drude auxiliary-state representation");
}

gpu::detail::lorentzian_internal_state_view
gpu::detail::lorentzian_internal_state_for_testing(
    void *internal_data, int component_value, int complex_part) {
  lorentzian_internal_state_view result = {
      nullptr, nullptr, 0, lorentzian_state_kind::previous};
  if (!internal_data || component_value < 0 ||
      component_value >= NUM_FIELD_COMPONENTS || complex_part < 0 ||
      complex_part >= 2)
    return result;
  lorentzian_data *data = static_cast<lorentzian_data *>(internal_data);
  result.polarization = data->P[component_value][complex_part];
  result.auxiliary = data->P_auxiliary[component_value][complex_part];
  result.point_count = data->ntot;
  result.kind = data->state_kind;
  return result;
}

// The second state has a susceptibility-lifetime role: it stores the latest
// increment for Drude poles and cancellation-prone FP32 Lorentz poles, and the
// previous polarization otherwise.  This preserves legacy behavior for
// well-conditioned coefficients without losing small restoring terms at tiny
// dt.
void *lorentzian_susceptibility::new_internal_data(realnum *W[NUM_FIELD_COMPONENTS][2],
                                                   const grid_volume &gv) const {
  int num = 0;
  FOR_COMPONENTS(c) DOCMP2 {
    if (needs_P(c, cmp, W)) num += 2 * gv.ntot();
  }
  size_t sz = sizeof(lorentzian_data) + sizeof(realnum) * (num - 1);
  lorentzian_data *d = (lorentzian_data *)malloc(sz);
  if (d == NULL) meep::abort("%s:%i:out of memory(%lu)", __FILE__, __LINE__, sz);
  d->sz_data = sz;
  return (void *)d;
}

void lorentzian_susceptibility::init_internal_data(realnum *W[NUM_FIELD_COMPONENTS][2], realnum dt,
                                                   const grid_volume &gv, void *data) const {
  lorentzian_data *d = (lorentzian_data *)data;
  size_t sz_data = d->sz_data;
  memset(d, 0, sz_data);
  d->sz_data = sz_data;
  d->state_kind = lorentzian_state_kind_for(
      no_omega_0_denominator, omega_0, gamma, dt);
  size_t ntot = d->ntot = gv.ntot();
  realnum *P = d->data;
  realnum *P_auxiliary = d->data + ntot;
  FOR_COMPONENTS(c) DOCMP2 {
    if (needs_P(c, cmp, W)) {
      d->P[c][cmp] = P;
      d->P_auxiliary[c][cmp] = P_auxiliary;
      P += 2 * ntot;
      P_auxiliary += 2 * ntot;
    }
  }
}

void *lorentzian_susceptibility::copy_internal_data(void *data) const {
  lorentzian_data *d = (lorentzian_data *)data;
  if (!d) return 0;
  require_valid_lorentzian_state_kind(d);
  lorentzian_data *dnew = (lorentzian_data *)malloc(d->sz_data);
  memcpy(dnew, d, d->sz_data);
  size_t ntot = d->ntot;
  realnum *P = dnew->data;
  realnum *P_auxiliary = dnew->data + ntot;
  FOR_COMPONENTS(c) DOCMP2 {
    if (d->P[c][cmp]) {
      dnew->P[c][cmp] = P;
      dnew->P_auxiliary[c][cmp] = P_auxiliary;
      P += 2 * ntot;
      P_auxiliary += 2 * ntot;
    }
  }
  return (void *)dnew;
}

#if 0
/* Return true if the discretized Lorentzian ODE is intrinsically unstable,
   i.e. if it corresponds to a filter with a pole z outside the unit circle.
   Note that the pole satisfies the quadratic equation:
            (z + 1/z - 2)/dt^2 + g*(z - 1/z)/(2*dt) + w^2 = 0
   where w = 2*pi*omega_0 and g = 2*pi*gamma.   It is just a little
   algebra from this to get the condition for a root with |z| > 1.

   FIXME: this test seems to be too conservative (issue #12) */
static bool lorentzian_unstable(realnum omega_0, realnum gamma, realnum dt) {
  realnum w = 2 * pi * omega_0, g = 2 * pi * gamma;
  realnum g2 = g * dt / 2, w2 = (w * dt) * (w * dt);
  realnum b = (1 - w2 / 2) / (1 + g2), c = (1 - g2) / (1 + g2);
  return b * b > c && 2 * b * b - c + 2 * fabs(b) * sqrt(b * b - c) > 1;
}
#endif

#define SWAP(t, a, b)                                                                              \
  {                                                                                                \
    t SWAP_temp = a;                                                                               \
    a = b;                                                                                         \
    b = SWAP_temp;                                                                                 \
  }

// stable averaging of offdiagonal components
#define OFFDIAG(u, g, sx, s)                                                                       \
  (0.25 * ((g[i] + g[i - sx]) * u[i] + (g[i + s] + g[(i + s) - sx]) * u[i + s]))

void lorentzian_susceptibility::update_P(realnum *W[NUM_FIELD_COMPONENTS][2],
                                         realnum *W_prev[NUM_FIELD_COMPONENTS][2], realnum dt,
                                         const grid_volume &gv, void *P_internal_data) const {
  lorentzian_data *d = (lorentzian_data *)P_internal_data;
  require_lorentzian_state_kind(
      d, no_omega_0_denominator, omega_0, gamma, dt);
  const realnum omega2pi = 2 * pi * omega_0, g2pi = gamma * 2 * pi;
  const realnum omega0dtsqr = omega2pi * omega2pi * dt * dt;
  const realnum gamma1inv = 1 / (1 + g2pi * dt / 2), gamma1 = (1 - g2pi * dt / 2);
  const realnum omega0dtsqr_denom = no_omega_0_denominator ? 0 : omega0dtsqr;
  (void)W_prev; // unused;

  // TODO: add back lorentzian_unstable(omega_0, gamma, dt) if we can improve the stability test

  FOR_COMPONENTS(c) DOCMP2 {
    if (d->P[c][cmp]) {
      const realnum *w = W[c][cmp], *s = sigma[c][component_direction(c)];
      if (w && s) {
        realnum *p = d->P[c][cmp], *aux = d->P_auxiliary[c][cmp];
        const bool increment_state =
            d->state_kind == gpu::detail::lorentzian_state_kind::increment;

        // directions/strides for offdiagonal terms, similar to update_eh
        const direction d = component_direction(c);
        const ptrdiff_t is = gv.stride(d) * (is_magnetic(c) ? -1 : +1);
        direction d1 = cycle_direction(gv.dim, d, 1);
        component c1 = direction_component(c, d1);
        ptrdiff_t is1 = gv.stride(d1) * (is_magnetic(c) ? -1 : +1);
        const realnum *w1 = W[c1][cmp];
        const realnum *s1 = w1 ? sigma[c][d1] : NULL;
        direction d2 = cycle_direction(gv.dim, d, 2);
        component c2 = direction_component(c, d2);
        ptrdiff_t is2 = gv.stride(d2) * (is_magnetic(c) ? -1 : +1);
        const realnum *w2 = W[c2][cmp];
        const realnum *s2 = w2 ? sigma[c][d2] : NULL;

        if (s2 && !s1) { // make s1 the non-NULL one if possible
          SWAP(direction, d1, d2);
          SWAP(component, c1, c2);
          SWAP(ptrdiff_t, is1, is2);
          SWAP(const realnum *, w1, w2);
          SWAP(const realnum *, s1, s2);
        }
        if (s1 && s2) { // 3x3 anisotropic
          PLOOP_OVER_VOL_OWNED(gv, c, i) {
            // s[i] != 0 check is a bit of a hack to work around
            // some instabilities that occur near the boundaries
            // of materials; see PR #666
            if (s[i] != 0) {
              if (increment_state) {
                const realnum dpnew =
                    gamma1inv * (gamma1 * aux[i] - omega0dtsqr_denom * p[i] +
                                 omega0dtsqr *
                                     (s[i] * w[i] + OFFDIAG(s1, w1, is1, is) +
                                      OFFDIAG(s2, w2, is2, is)));
                p[i] += dpnew;
                aux[i] = dpnew;
              }
              else {
                const realnum pcur = p[i];
                p[i] = gamma1inv *
                       (pcur * (2 - omega0dtsqr_denom) - gamma1 * aux[i] +
                        omega0dtsqr *
                            (s[i] * w[i] + OFFDIAG(s1, w1, is1, is) +
                             OFFDIAG(s2, w2, is2, is)));
                aux[i] = pcur;
              }
            }
          }
        }
        else if (s1) { // 2x2 anisotropic
          PLOOP_OVER_VOL_OWNED(gv, c, i) {
            if (s[i] != 0) { // see above
              if (increment_state) {
                const realnum dpnew =
                    gamma1inv * (gamma1 * aux[i] - omega0dtsqr_denom * p[i] +
                                 omega0dtsqr *
                                     (s[i] * w[i] + OFFDIAG(s1, w1, is1, is)));
                p[i] += dpnew;
                aux[i] = dpnew;
              }
              else {
                const realnum pcur = p[i];
                p[i] = gamma1inv *
                       (pcur * (2 - omega0dtsqr_denom) - gamma1 * aux[i] +
                        omega0dtsqr *
                            (s[i] * w[i] + OFFDIAG(s1, w1, is1, is)));
                aux[i] = pcur;
              }
            }
          }
        }
        else { // isotropic
          PLOOP_OVER_VOL_OWNED(gv, c, i) {
            if (increment_state) {
              const realnum dpnew =
                  gamma1inv * (gamma1 * aux[i] - omega0dtsqr_denom * p[i] +
                               omega0dtsqr * (s[i] * w[i]));
              p[i] += dpnew;
              aux[i] = dpnew;
            }
            else {
              const realnum pcur = p[i];
              p[i] = gamma1inv *
                     (pcur * (2 - omega0dtsqr_denom) - gamma1 * aux[i] +
                      omega0dtsqr * (s[i] * w[i]));
              aux[i] = pcur;
            }
          }
        }
      }
    }
  }
}

bool lorentzian_susceptibility::update_P_cuda(
    gpu::detail::resident_cache *cache,
    realnum *W[NUM_FIELD_COMPONENTS][2],
    realnum *W_prev[NUM_FIELD_COMPONENTS][2], realnum dt,
    const grid_volume &gv, void *P_internal_data) const {
#if MEEP_HAVE_CUDA
  (void)W_prev;
  if (!cache || !P_internal_data || sizeof(realnum) != sizeof(float))
    return false;

  lorentzian_data *d = static_cast<lorentzian_data *>(P_internal_data);
  require_lorentzian_state_kind(
      d, no_omega_0_denominator, omega_0, gamma, dt);
  const realnum omega2pi = 2 * pi * omega_0;
  const realnum g2pi = gamma * 2 * pi;
  const realnum omega0dtsqr = omega2pi * omega2pi * dt * dt;
  const gpu::detail::lorentzian_material_fp32 base_material = {
      nullptr,
      nullptr,
      nullptr,
      static_cast<float>(1 / (1 + g2pi * dt / 2)),
      static_cast<float>(1 - g2pi * dt / 2),
      static_cast<float>(omega0dtsqr),
      static_cast<float>(no_omega_0_denominator ? 0 : omega0dtsqr)};

  FOR_COMPONENTS(c) DOCMP2 {
    if (!d->P[c][cmp]) continue;
    const realnum *w = W[c][cmp];
    const direction dc = component_direction(c);
    const realnum *s = sigma[c][dc];
    if (!w || !s) continue;

    const ptrdiff_t field_stride =
        gv.stride(dc) * (is_magnetic(c) ? -1 : +1);
    const direction d1 = cycle_direction(gv.dim, dc, 1);
    const component c1 = direction_component(c, d1);
    const ptrdiff_t stride1 =
        gv.stride(d1) * (is_magnetic(c) ? -1 : +1);
    const realnum *w1 = W[c1][cmp];
    const realnum *s1 = w1 ? sigma[c][d1] : nullptr;
    const direction d2 = cycle_direction(gv.dim, dc, 2);
    const component c2 = direction_component(c, d2);
    const ptrdiff_t stride2 =
        gv.stride(d2) * (is_magnetic(c) ? -1 : +1);
    const realnum *w2 = W[c2][cmp];
    const realnum *s2 = w2 ? sigma[c][d2] : nullptr;

    const gpu::detail::index_space_fp32 index_space =
        gpu::detail::make_index_space_fp32(
            gv, gv.little_owned_corner(c), gv.big_corner());

    gpu::detail::lorentzian_material_fp32 material = base_material;
    material.sigma = s;
    material.offdiagonal1 = s1;
    material.offdiagonal2 = s2;
    gpu::detail::resident_update_lorentzian_fp32(
        cache, d->P[c][cmp], d->P_auxiliary[c][cmp], w, w1, w2,
        gv.ntot(), index_space, field_stride, stride1, stride2, material,
        d->state_kind);
  }
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

bool lorentzian_susceptibility::subtract_P_cuda(
    gpu::detail::resident_cache *cache, field_type ft,
    realnum *f_minus_p[NUM_FIELD_COMPONENTS][2],
    void *P_internal_data) const {
#if MEEP_HAVE_CUDA
  if (!cache || !P_internal_data || sizeof(realnum) != sizeof(float))
    return false;
  lorentzian_data *d = static_cast<lorentzian_data *>(P_internal_data);
  const field_type ft2 = ft == E_stuff ? D_stuff : B_stuff;
  FOR_FT_COMPONENTS(ft, ec) DOCMP2 {
    if (!d->P[ec][cmp]) continue;
    const component dc = field_type_component(ft2, ec);
    if (f_minus_p[dc][cmp])
      gpu::detail::resident_subtract_fp32(
          cache, f_minus_p[dc][cmp], d->P[ec][cmp], d->ntot);
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

void lorentzian_susceptibility::subtract_P(field_type ft,
                                           realnum *f_minus_p[NUM_FIELD_COMPONENTS][2],
                                           void *P_internal_data) const {
  lorentzian_data *d = (lorentzian_data *)P_internal_data;
  field_type ft2 = ft == E_stuff ? D_stuff : B_stuff; // for sources etc.
  size_t ntot = d->ntot;
  FOR_FT_COMPONENTS(ft, ec) DOCMP2 {
    if (d->P[ec][cmp]) {
      component dc = field_type_component(ft2, ec);
      if (f_minus_p[dc][cmp]) {
        realnum *p = d->P[ec][cmp];
        realnum *fmp = f_minus_p[dc][cmp];
        for (size_t i = 0; i < ntot; ++i)
          fmp[i] -= p[i];
      }
    }
  }
}

int lorentzian_susceptibility::num_cinternal_notowned_needed(component c,
                                                             void *P_internal_data) const {
  lorentzian_data *d = (lorentzian_data *)P_internal_data;
  return d->P[c][0] ? 1 : 0;
}

realnum *lorentzian_susceptibility::cinternal_notowned_ptr(int inotowned, component c, int cmp,
                                                           int n, void *P_internal_data) const {
  lorentzian_data *d = (lorentzian_data *)P_internal_data;
  (void)inotowned; // always = 0
  if (!d || !d->P[c][cmp]) return NULL;
  return d->P[c][cmp] + n;
}

std::complex<realnum> lorentzian_susceptibility::chi1(realnum freq, realnum sigma) {
  if (no_omega_0_denominator) {
    // Drude model
    return sigma * omega_0 * omega_0 / std::complex<realnum>(-freq * freq, -gamma * freq);
  }
  else {
    // Standard Lorentzian model
    return sigma * omega_0 * omega_0 /
           std::complex<realnum>(omega_0 * omega_0 - freq * freq, -gamma * freq);
  }
}

void lorentzian_susceptibility::dump_params(h5file *h5f, size_t *start) {
  size_t num_params = 5;
  size_t params_dims[1] = {num_params};
  realnum params_data[] = {4, (realnum)get_id(), omega_0, gamma, (realnum)no_omega_0_denominator};
  h5f->write_chunk(1, start, params_dims, params_data);
  *start += num_params;
}

void noisy_lorentzian_susceptibility::update_P(realnum *W[NUM_FIELD_COMPONENTS][2],
                                               realnum *W_prev[NUM_FIELD_COMPONENTS][2], realnum dt,
                                               const grid_volume &gv, void *P_internal_data) const {
  lorentzian_susceptibility::update_P(W, W_prev, dt, gv, P_internal_data);
  lorentzian_data *d = (lorentzian_data *)P_internal_data;

  const realnum g2pi = gamma * 2 * pi;
  const realnum w2pi = omega_0 * 2 * pi;
  const realnum amp = w2pi * noise_amp * sqrt(g2pi) * dt * dt / (1 + g2pi * dt / 2);
  /* for uniform random numbers in [-amp,amp] below, multiply amp by sqrt(3) */

  FOR_COMPONENTS(c) DOCMP2 {
    if (d->P[c][cmp]) {
      const realnum *s = sigma[c][component_direction(c)];
      if (s) {
        realnum *p = d->P[c][cmp];
        realnum *aux = d->P_auxiliary[c][cmp];
        LOOP_OVER_VOL_OWNED(gv, c, i) {
          const realnum increment = gaussian_random(0, amp * sqrt(s[i]));
          p[i] += increment;
          if (d->state_kind ==
              gpu::detail::lorentzian_state_kind::increment)
            aux[i] += increment;
        }
        // for uniform random numbers, use uniform_random(-1,1) * amp * sqrt(s[i])
        // for gaussian random numbers, use gaussian_random(0, amp * sqrt(s[i]))
      }
    }
  }
}

void noisy_lorentzian_susceptibility::dump_params(h5file *h5f, size_t *start) {
  size_t num_params = 6;
  size_t params_dims[1] = {num_params};
  realnum params_data[] = {
      5, (realnum)get_id(), noise_amp, omega_0, gamma, (realnum)no_omega_0_denominator};
  h5f->write_chunk(1, start, params_dims, params_data);
  *start += num_params;
}

gyrotropic_susceptibility::gyrotropic_susceptibility(const vec &bias, realnum omega_0,
                                                     realnum gamma, realnum alpha,
                                                     gyrotropy_model model)
    : omega_0(omega_0), gamma(gamma), alpha(alpha), model(model) {
  // Precalculate g_{ij} = sum_k epsilon_{ijk} b_k, used in update_P.
  // Ignore |b| for Landau-Lifshitz-Gilbert gyrotropy model.
  const vec b = (model == GYROTROPIC_SATURATED) ? bias / abs(bias) : bias;
  memset(gyro_tensor, 0, 9 * sizeof(realnum));
  gyro_tensor[X][Y] = b.z();
  gyro_tensor[Y][X] = -b.z();
  gyro_tensor[Y][Z] = b.x();
  gyro_tensor[Z][Y] = -b.x();
  gyro_tensor[Z][X] = b.y();
  gyro_tensor[X][Z] = -b.y();
}

/* To implement gyrotropic susceptibilities, we track three
   polarization components (e.g. Px, Py, Pz) on EACH of the Yee cell's
   three driving field positions (e.g., Ex, Ey, and Ez), i.e. 9
   numbers per cell.  This takes 3X the memory and runtime compared to
   Lorentzian susceptibility.  The advantage is that during update_P,
   we can directly access the value of P at each update point without
   averaging.  */

typedef struct {
  size_t sz_data;
  size_t ntot;
  realnum *P[NUM_FIELD_COMPONENTS][2][3];
  realnum *P_prev[NUM_FIELD_COMPONENTS][2][3];
  realnum data[1];
} gyrotropy_data;

void *gyrotropic_susceptibility::new_internal_data(realnum *W[NUM_FIELD_COMPONENTS][2],
                                                   const grid_volume &gv) const {
  int num = 0;
  FOR_COMPONENTS(c) DOCMP2 {
    if (needs_P(c, cmp, W)) num += 6 * gv.ntot();
  }
  size_t sz = sizeof(gyrotropy_data) + sizeof(realnum) * (num - 1);
  gyrotropy_data *d = (gyrotropy_data *)malloc(sz);
  if (d == NULL) meep::abort("%s:%i:out of memory(%lu)", __FILE__, __LINE__, sz);
  d->sz_data = sz;
  return (void *)d;
}

void gyrotropic_susceptibility::init_internal_data(realnum *W[NUM_FIELD_COMPONENTS][2], realnum dt,
                                                   const grid_volume &gv, void *data) const {
  (void)dt; // unused
  gyrotropy_data *d = (gyrotropy_data *)data;
  size_t sz_data = d->sz_data;
  memset(d, 0, sz_data);
  d->sz_data = sz_data;
  d->ntot = gv.ntot();
  realnum *p = d->data;
  FOR_COMPONENTS(c) DOCMP2 {
    if (needs_P(c, cmp, W)) {
      for (int dd = X; dd < R; dd++) {
        d->P[c][cmp][dd] = p;
        p += d->ntot;
        d->P_prev[c][cmp][dd] = p;
        p += d->ntot;
      }
    }
  }
}

void *gyrotropic_susceptibility::copy_internal_data(void *data) const {
  gyrotropy_data *d = (gyrotropy_data *)data;
  if (!d) return 0;
  gyrotropy_data *dnew = (gyrotropy_data *)malloc(d->sz_data);
  memcpy(dnew, d, d->sz_data);
  realnum *p = dnew->data;
  FOR_COMPONENTS(c) DOCMP2 {
    if (d->P[c][cmp][0]) {
      for (int dd = X; dd < R; dd++) {
        dnew->P[c][cmp][dd] = p;
        p += d->ntot;
        dnew->P_prev[c][cmp][dd] = p;
        p += d->ntot;
      }
    }
  }
  return (void *)dnew;
}

bool gyrotropic_susceptibility::needs_P(component c, int cmp,
                                        realnum *W[NUM_FIELD_COMPONENTS][2]) const {
  if (!is_electric(c) && !is_magnetic(c)) return false;
  direction d0 = component_direction(c);
  return (d0 == X || d0 == Y || d0 == Z) && sigma[c][d0] && W[c][cmp];
}

// Similar to the OFFDIAG macro, but without averaging sigma.
#define OFFDIAGW(g, sx, s) (0.25 * (g[i] + g[i - sx] + g[i + s] + g[i + s - sx]))

void gyrotropic_susceptibility::update_P(realnum *W[NUM_FIELD_COMPONENTS][2],
                                         realnum *W_prev[NUM_FIELD_COMPONENTS][2], realnum dt,
                                         const grid_volume &gv, void *P_internal_data) const {
  gyrotropy_data *d = (gyrotropy_data *)P_internal_data;
  const realnum omega2pidt = 2 * pi * omega_0 * dt;
  const realnum g2pidt = 2 * pi * gamma * dt;
  (void)W_prev; // unused;

  switch (model) {
    case GYROTROPIC_LORENTZIAN:
    case GYROTROPIC_DRUDE: {
      const realnum omega0dtsqr = omega2pidt * omega2pidt;
      const realnum gamma1 = (1 - g2pidt / 2);
      const realnum diag = 2 - (model == GYROTROPIC_DRUDE ? 0 : omega0dtsqr);
      const realnum pt = pi * dt;

      // Precalculate 3x3 matrix inverse, exploiting skew symmetry
      const realnum gd = (1 + g2pidt / 2);
      const realnum gx = pt * gyro_tensor[Y][Z];
      const realnum gy = pt * gyro_tensor[Z][X];
      const realnum gz = pt * gyro_tensor[X][Y];
      const realnum invdet = 1.0 / gd / (gd * gd + gx * gx + gy * gy + gz * gz);
      const realnum inv[3][3] = {{invdet * (gd * gd + gx * gx), invdet * (gx * gy + gd * gz),
                                  invdet * (gx * gz - gd * gy)},
                                 {invdet * (gy * gx - gd * gz), invdet * (gd * gd + gy * gy),
                                  invdet * (gy * gz + gd * gx)},
                                 {invdet * (gz * gx + gd * gy), invdet * (gz * gy - gd * gx),
                                  invdet * (gd * gd + gz * gz)}};

      FOR_COMPONENTS(c) DOCMP2 {
        if (d->P[c][cmp][0]) {
          const direction d0 = component_direction(c);
          const realnum *w0 = W[c][cmp], *s = sigma[c][d0];

          if (!w0 || !s || (d0 != X && d0 != Y && d0 != Z))
            meep::abort("gyrotropic media require 3D Cartesian fields\n");

          const direction d1 = cycle_direction(gv.dim, d0, 1);
          const direction d2 = cycle_direction(gv.dim, d0, 2);
          const realnum *w1 = W[direction_component(c, d1)][cmp];
          const realnum *w2 = W[direction_component(c, d2)][cmp];
          realnum *p0 = d->P[c][cmp][d0], *pp0 = d->P_prev[c][cmp][d0];
          realnum *p1 = d->P[c][cmp][d1], *pp1 = d->P_prev[c][cmp][d1];
          realnum *p2 = d->P[c][cmp][d2], *pp2 = d->P_prev[c][cmp][d2];
          const ptrdiff_t is = gv.stride(d0) * (is_magnetic(c) ? -1 : +1);
          const ptrdiff_t is1 = gv.stride(d1) * (is_magnetic(c) ? -1 : +1);
          const ptrdiff_t is2 = gv.stride(d2) * (is_magnetic(c) ? -1 : +1);
          realnum r0, r1, r2;

          if (!pp1 || !pp2) meep::abort("gyrotropic media require 3D Cartesian fields\n");
          if (sigma[c][d1] || sigma[c][d2])
            meep::abort("gyrotropic media do not support anisotropic sigma\n");

          LOOP_OVER_VOL_OWNED(gv, c, i) {
            r0 = diag * p0[i] - gamma1 * pp0[i] + omega0dtsqr * s[i] * w0[i] -
                 pt * gyro_tensor[d0][d1] * pp1[i] - pt * gyro_tensor[d0][d2] * pp2[i];
            r1 = diag * p1[i] - gamma1 * pp1[i] +
                 (w1 ? omega0dtsqr * s[i] * OFFDIAGW(w1, is1, is) : 0) -
                 pt * gyro_tensor[d1][d0] * pp0[i] - pt * gyro_tensor[d1][d2] * pp2[i];
            r2 = diag * p2[i] - gamma1 * pp2[i] +
                 (w2 ? omega0dtsqr * s[i] * OFFDIAGW(w2, is2, is) : 0) -
                 pt * gyro_tensor[d2][d1] * pp1[i] - pt * gyro_tensor[d2][d0] * pp0[i];

            pp0[i] = p0[i];
            pp1[i] = p1[i];
            pp2[i] = p2[i];
            p0[i] = inv[d0][d0] * r0 + inv[d0][d1] * r1 + inv[d0][d2] * r2;
            p1[i] = inv[d1][d0] * r0 + inv[d1][d1] * r1 + inv[d1][d2] * r2;
            p2[i] = inv[d2][d0] * r0 + inv[d2][d1] * r1 + inv[d2][d2] * r2;
          }
        }
      }
    } break;

    case GYROTROPIC_SATURATED: {
      const realnum dt2pi = 2 * pi * dt;

      // Precalculate 3x3 matrix inverse, exploiting skew symmetry
      const realnum gd = 0.5;
      const realnum gx = -0.5 * alpha * gyro_tensor[Y][Z];
      const realnum gy = -0.5 * alpha * gyro_tensor[Z][X];
      const realnum gz = -0.5 * alpha * gyro_tensor[X][Y];
      const realnum invdet = 1.0 / gd / (gd * gd + gx * gx + gy * gy + gz * gz);
      const realnum inv[3][3] = {{invdet * (gd * gd + gx * gx), invdet * (gx * gy + gd * gz),
                                  invdet * (gx * gz - gd * gy)},
                                 {invdet * (gy * gx - gd * gz), invdet * (gd * gd + gy * gy),
                                  invdet * (gy * gz + gd * gx)},
                                 {invdet * (gz * gx + gd * gy), invdet * (gz * gy - gd * gx),
                                  invdet * (gd * gd + gz * gz)}};

      FOR_COMPONENTS(c) DOCMP2 {
        if (d->P[c][cmp][0]) {
          const direction d0 = component_direction(c);
          const realnum *w0 = W[c][cmp], *s = sigma[c][d0];

          if (!w0 || !s || (d0 != X && d0 != Y && d0 != Z))
            meep::abort("gyrotropic media require 3D Cartesian fields\n");

          const direction d1 = cycle_direction(gv.dim, d0, 1);
          const direction d2 = cycle_direction(gv.dim, d0, 2);
          const realnum *w1 = W[direction_component(c, d1)][cmp];
          const realnum *w2 = W[direction_component(c, d2)][cmp];
          realnum *p0 = d->P[c][cmp][d0], *pp0 = d->P_prev[c][cmp][d0];
          realnum *p1 = d->P[c][cmp][d1], *pp1 = d->P_prev[c][cmp][d1];
          realnum *p2 = d->P[c][cmp][d2], *pp2 = d->P_prev[c][cmp][d2];
          const ptrdiff_t is = gv.stride(d0) * (is_magnetic(c) ? -1 : +1);
          const ptrdiff_t is1 = gv.stride(d1) * (is_magnetic(c) ? -1 : +1);
          const ptrdiff_t is2 = gv.stride(d2) * (is_magnetic(c) ? -1 : +1);
          realnum r0, r1, r2, q0, q1, q2;

          if (!pp1 || !pp2) meep::abort("gyrotropic media require 3D Cartesian fields\n");
          if (sigma[c][d1] || sigma[c][d2])
            meep::abort("gyrotropic media do not support anisotropic sigma\n");

          LOOP_OVER_VOL_OWNED(gv, c, i) {
            q0 = -omega2pidt * p0[i] + 0.5 * alpha * pp0[i] + dt2pi * s[i] * w0[i];
            q1 = -omega2pidt * p1[i] + 0.5 * alpha * pp1[i] +
                 dt2pi * s[i] * (w1 ? OFFDIAGW(w1, is1, is) : 0);
            q2 = -omega2pidt * p2[i] + 0.5 * alpha * pp2[i] +
                 dt2pi * s[i] * (w2 ? OFFDIAGW(w2, is2, is) : 0);

            r0 =
                0.5 * pp0[i] - g2pidt * p0[i] + gyro_tensor[d0][d1] * q1 + gyro_tensor[d0][d2] * q2;
            r1 =
                0.5 * pp1[i] - g2pidt * p1[i] + gyro_tensor[d1][d2] * q2 + gyro_tensor[d1][d0] * q0;
            r2 =
                0.5 * pp2[i] - g2pidt * p2[i] + gyro_tensor[d2][d0] * q0 + gyro_tensor[d2][d1] * q1;

            pp0[i] = p0[i];
            pp1[i] = p1[i];
            pp2[i] = p2[i];
            p0[i] = inv[d0][d0] * r0 + inv[d0][d1] * r1 + inv[d0][d2] * r2;
            p1[i] = inv[d1][d0] * r0 + inv[d1][d1] * r1 + inv[d1][d2] * r2;
            p2[i] = inv[d2][d0] * r0 + inv[d2][d1] * r1 + inv[d2][d2] * r2;
          }
        }
      }
    } break;
  }
}

bool gyrotropic_susceptibility::cuda_update_P_eligible(
    realnum *W[NUM_FIELD_COMPONENTS][2], const grid_volume &gv,
    std::string *reason) const {
  const auto reject = [reason](const char *message) {
    if (reason) *reason = message;
    return false;
  };
  if (sizeof(realnum) != sizeof(float))
    return reject("gyrotropic CUDA stepping requires FP32 fields");
  switch (model) {
    case GYROTROPIC_LORENTZIAN:
    case GYROTROPIC_DRUDE:
    case GYROTROPIC_SATURATED: break;
    default:
      return reject("gyrotropic CUDA stepping received an invalid model");
  }
  FOR_COMPONENTS(c) DOCMP2 {
    if (!needs_P(c, cmp, W)) continue;
    const direction d0 = component_direction(c);
    if ((d0 != X && d0 != Y && d0 != Z) || !W[c][cmp] ||
        !sigma[c][d0])
      return reject(
          "gyrotropic CUDA stepping requires Cartesian primary fields");
    const direction d1 = cycle_direction(gv.dim, d0, 1);
    const direction d2 = cycle_direction(gv.dim, d0, 2);
    if (sigma[c][d1] || sigma[c][d2])
      return reject(
          "gyrotropic CUDA stepping does not support anisotropic sigma");
  }
  if (reason) reason->clear();
  return true;
}

bool gyrotropic_susceptibility::update_P_cuda(
    gpu::detail::resident_cache *cache,
    realnum *W[NUM_FIELD_COMPONENTS][2],
    realnum *W_prev[NUM_FIELD_COMPONENTS][2], realnum dt,
    const grid_volume &gv, void *P_internal_data) const {
#if MEEP_HAVE_CUDA
  (void)W_prev;
  if (!cache || !P_internal_data || sizeof(realnum) != sizeof(float))
    return false;
  std::string eligibility_reason;
  if (!cuda_update_P_eligible(W, gv, &eligibility_reason))
    throw std::runtime_error(eligibility_reason);

  gyrotropy_data *d =
      static_cast<gyrotropy_data *>(P_internal_data);
  const realnum omega2pidt = 2 * pi * omega_0 * dt;
  const realnum g2pidt = 2 * pi * gamma * dt;
  const bool saturated = model == GYROTROPIC_SATURATED;
  const realnum gd = saturated ? 0.5 : 1 + g2pidt / 2;
  const realnum gx =
      (saturated ? -0.5 * alpha : pi * dt) * gyro_tensor[Y][Z];
  const realnum gy =
      (saturated ? -0.5 * alpha : pi * dt) * gyro_tensor[Z][X];
  const realnum gz =
      (saturated ? -0.5 * alpha : pi * dt) * gyro_tensor[X][Y];
  const realnum invdet =
      1.0 / gd / (gd * gd + gx * gx + gy * gy + gz * gz);
  const realnum inverse[3][3] = {
      {invdet * (gd * gd + gx * gx),
       invdet * (gx * gy + gd * gz),
       invdet * (gx * gz - gd * gy)},
      {invdet * (gy * gx - gd * gz),
       invdet * (gd * gd + gy * gy),
       invdet * (gy * gz + gd * gx)},
      {invdet * (gz * gx + gd * gy),
       invdet * (gz * gy - gd * gx),
       invdet * (gd * gd + gz * gz)}};

  // Validate the complete susceptibility before launching any component.
  // Required-CUDA mode must never leave a partially advanced gyrotropic
  // state if a later component has an unsupported layout.
  FOR_COMPONENTS(c) DOCMP2 {
    if (!d->P[c][cmp][0]) continue;
    const direction d0 = component_direction(c);
    if ((d0 != X && d0 != Y && d0 != Z) || !W[c][cmp] ||
        !sigma[c][d0])
      throw std::runtime_error(
          "CUDA gyrotropic media require Cartesian primary fields");
    const direction d1 = cycle_direction(gv.dim, d0, 1);
    const direction d2 = cycle_direction(gv.dim, d0, 2);
    if (!d->P[c][cmp][d1] || !d->P[c][cmp][d2])
      throw std::runtime_error(
          "CUDA gyrotropic media require three polarization components");
    if (sigma[c][d1] || sigma[c][d2])
      throw std::runtime_error(
          "CUDA gyrotropic media do not support anisotropic sigma");
  }

  FOR_COMPONENTS(c) DOCMP2 {
    if (!d->P[c][cmp][0]) continue;
    const direction d0 = component_direction(c);
    const direction d1 = cycle_direction(gv.dim, d0, 1);
    const direction d2 = cycle_direction(gv.dim, d0, 2);
    const component c1 = direction_component(c, d1);
    const component c2 = direction_component(c, d2);
    const direction directions[3] = {d0, d1, d2};

    gpu::detail::gyrotropic_material_fp32 material = {};
    material.sigma = sigma[c][d0];
    switch (model) {
      case GYROTROPIC_LORENTZIAN:
        material.model =
            meep_cuda::detail::gyrotropic_lorentzian_fp32;
        break;
      case GYROTROPIC_DRUDE:
        material.model = meep_cuda::detail::gyrotropic_drude_fp32;
        break;
      case GYROTROPIC_SATURATED:
        material.model =
            meep_cuda::detail::gyrotropic_saturated_fp32;
        break;
    }
    for (int row = 0; row < 3; ++row)
      for (int column = 0; column < 3; ++column) {
        const int entry = 3 * row + column;
        material.inverse[entry] = static_cast<float>(
            inverse[directions[row]][directions[column]]);
        material.gyro[entry] = static_cast<float>(
            gyro_tensor[directions[row]][directions[column]]);
      }
    if (saturated) {
      material.omega_dt = static_cast<float>(omega2pidt);
      material.gamma_dt = static_cast<float>(g2pidt);
      material.alpha_half = static_cast<float>(0.5 * alpha);
      material.drive_dt = static_cast<float>(2 * pi * dt);
    }
    else {
      const realnum omega_dt_squared = omega2pidt * omega2pidt;
      material.diagonal = static_cast<float>(
          2 - (model == GYROTROPIC_DRUDE ? 0 : omega_dt_squared));
      material.gamma_previous =
          static_cast<float>(1 - g2pidt / 2);
      material.omega_dt_squared =
          static_cast<float>(omega_dt_squared);
      material.precession_half_dt = static_cast<float>(pi * dt);
    }

    const ptrdiff_t field_stride =
        gv.stride(d0) * (is_magnetic(c) ? -1 : +1);
    const ptrdiff_t stride1 =
        gv.stride(d1) * (is_magnetic(c) ? -1 : +1);
    const ptrdiff_t stride2 =
        gv.stride(d2) * (is_magnetic(c) ? -1 : +1);
    const gpu::detail::index_space_fp32 index_space =
        gpu::detail::make_index_space_fp32(
            gv, gv.little_owned_corner(c), gv.big_corner());
    gpu::detail::resident_update_gyrotropic_fp32(
        cache, d->P[c][cmp][d0], d->P[c][cmp][d1],
        d->P[c][cmp][d2], d->P_prev[c][cmp][d0],
        d->P_prev[c][cmp][d1], d->P_prev[c][cmp][d2], W[c][cmp],
        W[c1][cmp], W[c2][cmp], gv.ntot(), index_space, field_stride,
        stride1, stride2, material);
  }
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

bool gyrotropic_susceptibility::subtract_P_cuda(
    gpu::detail::resident_cache *cache, field_type ft,
    realnum *f_minus_p[NUM_FIELD_COMPONENTS][2],
    void *P_internal_data) const {
#if MEEP_HAVE_CUDA
  if (!cache || !P_internal_data || sizeof(realnum) != sizeof(float))
    return false;
  gyrotropy_data *d =
      static_cast<gyrotropy_data *>(P_internal_data);
  const field_type ft2 = ft == E_stuff ? D_stuff : B_stuff;
  FOR_FT_COMPONENTS(ft, ec) DOCMP2 {
    if (!d->P[ec][cmp][0]) continue;
    const component dc = field_type_component(ft2, ec);
    if (f_minus_p[dc][cmp])
      gpu::detail::resident_subtract_fp32(
          cache, f_minus_p[dc][cmp],
          d->P[ec][cmp][component_direction(ec)], d->ntot);
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

void gyrotropic_susceptibility::subtract_P(field_type ft,
                                           realnum *f_minus_p[NUM_FIELD_COMPONENTS][2],
                                           void *P_internal_data) const {
  gyrotropy_data *d = (gyrotropy_data *)P_internal_data;
  field_type ft2 = ft == E_stuff ? D_stuff : B_stuff; // for sources etc.
  size_t ntot = d->ntot;
  FOR_FT_COMPONENTS(ft, ec) DOCMP2 {
    if (d->P[ec][cmp][0]) {
      component dc = field_type_component(ft2, ec);
      if (f_minus_p[dc][cmp]) {
        realnum *p = d->P[ec][cmp][component_direction(ec)];
        realnum *fmp = f_minus_p[dc][cmp];
        for (size_t i = 0; i < ntot; ++i)
          fmp[i] -= p[i];
      }
    }
  }
}

int gyrotropic_susceptibility::num_cinternal_notowned_needed(component c,
                                                             void *P_internal_data) const {
  (void)c;
  (void)P_internal_data;
  return 0;
}

realnum *gyrotropic_susceptibility::cinternal_notowned_ptr(int inotowned, component c, int cmp,
                                                           int n, void *P_internal_data) const {
  gyrotropy_data *d = (gyrotropy_data *)P_internal_data;
  if (!d || !d->P[c][cmp][inotowned]) return NULL;
  return d->P[c][cmp][inotowned] + n;
}

void gyrotropic_susceptibility::dump_params(h5file *h5f, size_t *start) {
  size_t num_params = 9;
  size_t params_dims[1] = {num_params};
  realnum bias[] = {gyro_tensor[Y][Z], gyro_tensor[Z][X], gyro_tensor[X][Y]};
  realnum params_data[] = {8,     (realnum)get_id(), bias[X], bias[Y], bias[Z], omega_0, gamma,
                           alpha, (realnum)model};
  h5f->write_chunk(1, start, params_dims, params_data);
  *start += num_params;
}

} // namespace meep
