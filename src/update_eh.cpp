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

#include <string.h>
#include <assert.h>

#include "meep.hpp"
#include "meep_internals.hpp"
#include "gpu_backend_internal.hpp"
#include "gpu_grid_index.hpp"

#include <limits>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <typeinfo>

using namespace std;

namespace meep {
namespace {

size_t update_point_count(const ivec &is, const ivec &ie) {
  size_t count = 1;
  for (int axis = 0; axis < 3; ++axis) {
    const ptrdiff_t extent =
        (ie.yucky_val(axis) - is.yucky_val(axis)) / 2 + 1;
    if (extent <= 0) return 0;
    if (count >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(extent))
      throw std::overflow_error("Meep E/H update point count overflow");
    count *= static_cast<size_t>(extent);
  }
  return count;
}

} // namespace

void fields::update_eh(field_type ft, bool skip_w_components) {
  if (ft != E_stuff && ft != H_stuff) meep::abort("update_eh only works with E/H");

  // split the chunks' volume into subdomains for tiled execution of update_eh loop
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine() && changed_materials) {
      bool is_aniso = false;
      FOR_FT_COMPONENTS(ft, cc) {
        const direction d_c = component_direction(cc);
        const direction d_1 = cycle_direction(chunks[i]->gv.dim, d_c, 1);
        const direction d_2 = cycle_direction(chunks[i]->gv.dim, d_c, 2);
        if (chunks[i]->s->chi1inv[cc][d_1] && chunks[i]->s->chi1inv[cc][d_2]) {
          is_aniso = true;
          break;
        }
      }
      if (!chunks[i]->gvs_eh[ft].empty()) chunks[i]->gvs_eh[ft].clear();
      if (loop_tile_base_eh > 0 && is_aniso) {
        split_into_tiles(chunks[i]->gv, &chunks[i]->gvs_eh[ft], loop_tile_base_eh);
        check_tiles(chunks[i]->gv, chunks[i]->gvs_eh[ft], loop_tile_base_eh);
      }
      else { chunks[i]->gvs_eh[ft].push_back(chunks[i]->gv); }
    }

  gpu::detail::phase_batch_mode phase_batch_mode =
      gpu::detail::cuda_active()
          ? gpu::detail::phase_batched_update_eh_mode()
          : gpu::detail::phase_batch_mode::disabled;
  for (int i = 0;
       i < num_chunks &&
       phase_batch_mode != gpu::detail::phase_batch_mode::disabled;
       ++i)
    if (chunks[i]->is_mine() &&
        !gpu::detail::resident_phase_is_active_for_owner(chunks[i]))
      phase_batch_mode = gpu::detail::phase_batch_mode::disabled;
  const int phase_key =
      2 * static_cast<int>(ft) + (skip_w_components ? 1 : 0);
  gpu::detail::resident_update_eh_phase_batch phase_batch(
      this, phase_key, phase_batch_mode);

  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine())
      if (chunks[i]->update_eh(ft, skip_w_components)) {
        chunk_connections_valid = false; // E/H allocated - reconnect chunks
        assert(changed_materials);
      }
  phase_batch.finish();
}

bool fields_chunk::needs_W_prev(component c) const {
  for (susceptibility *chiP = s->chiP[type(c)]; chiP; chiP = chiP->next)
    if (chiP->needs_W_prev()) return true;
  return false;
}

bool fields_chunk::cuda_whole_update_eh_overlap_eligible(
    field_type ft, std::string *reason) const {
  const auto reject = [reason](const char *message) {
    if (reason) *reason = message;
    return false;
  };
  if (ft != E_stuff && ft != H_stuff)
    return reject("overlap target is not E/H");
  if (gv.dim == Dcyl)
    return reject("cylindrical E/H update");
  if (doing_solve_cw)
    return reject("solve-cw E/H update");
  if (pol[ft] || s->chiP[ft])
    return reject("dispersive polarization");
  if (gvs_eh[ft].empty())
    return reject("uninitialized E/H update volumes");

  const field_type ft2 = ft == E_stuff ? D_stuff : B_stuff;
  for (const src_vol &source : sources[ft2])
    if (!source.t() || source.t()->is_integrated)
      return reject("integrated E/H source");

  bool have_update = false;
  FOR_FT_COMPONENTS(ft, ec) {
    const component dc = field_type_component(ft2, ec);
    const direction d_ec = component_direction(ec);
    const direction d_1 = cycle_direction(gv.dim, d_ec, 1);
    const direction d_2 = cycle_direction(gv.dim, d_ec, 2);
    if (s->chi1inv[ec][d_1] || s->chi1inv[ec][d_2])
      return reject("off-diagonal constitutive update");
    if (s->chi2[ec] || s->chi3[ec])
      return reject("nonlinear constitutive update");
    if (needs_W_prev(ec))
      return reject("previous E/H auxiliary field");

    const bool pml = s->sigsize[d_ec] > 1;
    const int component_count = is_real ? 1 : 2;
    for (int cmp = 0; cmp < component_count; ++cmp) {
      if (f_minus_p[dc][cmp])
        return reject("D/B-minus-polarization scratch");
      if (!f[ec][cmp]) continue;
      if (f[ec][cmp] == f[dc][cmp] &&
          (s->chi1inv[ec][d_ec] || pml))
        return reject("lazy E/H allocation");
      if (pml && !f_w[ec][cmp])
        return reject("lazy PML E/H auxiliary allocation");
      have_update = have_update || f[ec][cmp] != f[dc][cmp];
    }
  }
  if (!have_update)
    return reject("empty E/H update");
  if (reason) reason->clear();
  return true;
}

bool fields_chunk::update_eh(field_type ft, bool skip_w_components) {
  field_type ft2 = ft == E_stuff ? D_stuff : B_stuff; // for sources etc.
  bool allocated_eh = false;

  bool have_int_sources = false;
  if (!doing_solve_cw) {
    for (const src_vol &sv : sources[ft2]) {
      if (sv.t()->is_integrated) {
        have_int_sources = true;
        break;
      }
    }
  }

  FOR_FT_COMPONENTS(ft, ec) {
    component dc = field_type_component(ft2, ec);
    DOCMP {
      bool need_fmp = false;
      if (f[ec][cmp]) {
        need_fmp = have_int_sources;
        for (polarization_state *p = pol[ft]; p && !need_fmp; p = p->next)
          need_fmp = need_fmp || p->s->needs_P(ec, cmp, f);
      }
      if (need_fmp) {
        if (!f_minus_p[dc][cmp]) f_minus_p[dc][cmp] = new realnum[gv.ntot()];
      }
      else if (f_minus_p[dc][cmp]) { // remove unneeded f_minus_p
        // Resident CUDA mirrors are keyed by host address. Forget only this
        // disposable scratch mirror: destroying the whole cache here would
        // discard device-authoritative fields in an outer resident time step.
        gpu::detail::discard_resident_mirror_for_owner(
            this, f_minus_p[dc][cmp]);
        delete[] f_minus_p[dc][cmp];
        f_minus_p[dc][cmp] = 0;
      }
    }
  }
  bool have_f_minus_p = false;
  FOR_FT_COMPONENTS(ft2, dc) {
    if (f_minus_p[dc][0]) {
      have_f_minus_p = true;
      break;
    }
  }

  const size_t ntot = s->gv.ntot();

  if (have_f_minus_p && doing_solve_cw)
    meep::abort("dispersive materials are not yet implemented for solve_cw");

  const bool cuda_backend_active = gpu::detail::cuda_active();
  const bool cuda_phase_eligible =
      cuda_backend_active && sizeof(realnum) == sizeof(float);
  if (cuda_backend_active && !cuda_phase_eligible)
    throw std::runtime_error(
        "preflighted CUDA E/H update requires FP32 fields");
  gpu::detail::resident_curl_session cuda_session(this,
                                                   cuda_phase_eligible);
  const auto allocate_copy = [&](realnum *source) {
    std::unique_ptr<realnum[]> destination(new realnum[gv.ntot()]);
#if MEEP_HAVE_CUDA
    if (cuda_session.active()) {
      try {
        gpu::detail::resident_copy_fp32(
            cuda_session.cache(), destination.get(), source, gv.ntot());
      }
      catch (...) {
        gpu::detail::discard_resident_mirror_for_owner(
            this, destination.get());
        throw;
      }
    }
    else
#endif
      memcpy(destination.get(), source, gv.ntot() * sizeof(realnum));
    return destination.release();
  };

  //////////////////////////////////////////////////////////////////////////
  // First, initialize f_minus_p to D - P, if necessary

  bool cuda_f_minus_p = cuda_session.active();
#if MEEP_HAVE_CUDA
  for (polarization_state *p = pol[ft]; p && cuda_f_minus_p; p = p->next) {
    const lorentzian_susceptibility *lorentzian =
        dynamic_cast<const lorentzian_susceptibility *>(p->s);
    const bool standard_lorentzian =
        lorentzian &&
        typeid(*p->s) == typeid(lorentzian_susceptibility);
    const gyrotropic_susceptibility *gyrotropic =
        dynamic_cast<const gyrotropic_susceptibility *>(p->s);
    const bool standard_gyrotropic =
        gyrotropic &&
        typeid(*p->s) == typeid(gyrotropic_susceptibility);
    const multilevel_susceptibility *multilevel =
        dynamic_cast<const multilevel_susceptibility *>(p->s);
    const bool standard_multilevel =
        multilevel &&
        typeid(*p->s) == typeid(multilevel_susceptibility);
    cuda_f_minus_p =
        standard_lorentzian || standard_gyrotropic ||
        standard_multilevel;
  }
#else
  cuda_f_minus_p = false;
#endif

  if (!cuda_f_minus_p && cuda_session.active()) {
    throw std::runtime_error(
        "preflighted CUDA polarization subtraction supports only standard "
        "Lorentzian/Drude, gyrotropic, and multilevel "
        "susceptibilities");
  }

  if (cuda_f_minus_p) {
#if MEEP_HAVE_CUDA
    FOR_FT_COMPONENTS(ft, ec) if (f[ec][0]) {
      const component dc = field_type_component(ft2, ec);
      DOCMP if (f_minus_p[dc][cmp]) {
        gpu::detail::resident_copy_fp32(
            cuda_session.cache(), f_minus_p[dc][cmp], f[dc][cmp], ntot);
      }
    }

    for (polarization_state *p = pol[ft]; p; p = p->next)
      if (p->data) {
        const lorentzian_susceptibility *lorentzian =
            dynamic_cast<const lorentzian_susceptibility *>(p->s);
        const bool standard_lorentzian =
            lorentzian &&
            typeid(*p->s) == typeid(lorentzian_susceptibility);
        const gyrotropic_susceptibility *gyrotropic =
            dynamic_cast<const gyrotropic_susceptibility *>(p->s);
        const bool standard_gyrotropic =
            gyrotropic &&
            typeid(*p->s) == typeid(gyrotropic_susceptibility);
        const multilevel_susceptibility *multilevel =
            dynamic_cast<const multilevel_susceptibility *>(p->s);
        const bool standard_multilevel =
            multilevel &&
            typeid(*p->s) == typeid(multilevel_susceptibility);
        bool subtracted = false;
        if (standard_lorentzian)
          subtracted = lorentzian->subtract_P_cuda(
              cuda_session.cache(), ft, f_minus_p, p->data);
        else if (standard_gyrotropic)
          subtracted = gyrotropic->subtract_P_cuda(
              cuda_session.cache(), ft, f_minus_p, p->data);
        else if (standard_multilevel)
          subtracted = multilevel->subtract_P_cuda(
              cuda_session.cache(), ft, f_minus_p, p->data);
        if (!subtracted)
          throw std::logic_error(
              "preflighted CUDA polarization subtraction was not dispatched");
      }
#endif
  }
  else {
    FOR_FT_COMPONENTS(ft, ec) if (f[ec][0]) {
      component dc = field_type_component(ft2, ec);
      DOCMP if (f_minus_p[dc][cmp]) {
        realnum *fmp = f_minus_p[dc][cmp];
        memcpy(fmp, f[dc][cmp], sizeof(realnum) * ntot);
      }
    }

    for (polarization_state *p = pol[ft]; p; p = p->next)
      if (p->data) p->s->subtract_P(ft, f_minus_p, p->data);
  }

  //////////////////////////////////////////////////////////////////////////
  // Next, subtract time-integrated sources (i.e. polarizations, not currents)

  gpu::detail::resident_source_phase_batch integrated_source_batch(
      this, static_cast<int>(ft),
      have_f_minus_p && !doing_solve_cw && cuda_f_minus_p &&
          gpu::detail::phase_batched_source_opted_in());
  if (have_f_minus_p && !doing_solve_cw && cuda_f_minus_p) {
#if MEEP_HAVE_CUDA
    for (const src_vol &sv : sources[ft2]) {
      if (sv.t()->is_integrated && f[sv.c][0] && ft == type(sv.c)) {
        const component c = field_type_component(ft2, sv.c);
        const complex<double> time_scale = sv.t()->dipole();
        bool phase_batched = false;
        DOCMP {
          phase_batched =
              gpu::detail::resident_indexed_source_subtract_fp32(
                  cuda_session.cache(), f_minus_p[c][cmp], ntot,
                  sv.indices_data(), sv.amplitudes_fp32_data(),
                  sv.num_points(), nullptr,
                  {static_cast<float>(time_scale.real()),
                   static_cast<float>(time_scale.imag())},
                  cmp != 0) ||
              phase_batched;
        }
        if (!phase_batched)
          gpu::detail::record_cuda_source(
              sv.num_points() * static_cast<size_t>(is_real ? 1 : 2));
      }
    }
#endif
  }
  else if (have_f_minus_p && !doing_solve_cw) {
    for (const src_vol &sv : sources[ft2]) {
      if (sv.t()->is_integrated && f[sv.c][0] && ft == type(sv.c)) {
        component c = field_type_component(ft2, sv.c);
        for (size_t j = 0; j < sv.num_points(); ++j) {
          const complex<double> A = sv.dipole(j);
          DOCMP { f_minus_p[c][cmp][sv.index_at(j)] -= (cmp) ? imag(A) : real(A); }
        }
        gpu::detail::record_cpu_source(
            sv.num_points() * static_cast<size_t>(is_real ? 1 : 2));
      }
    }
  }
  integrated_source_batch.finish();

  //////////////////////////////////////////////////////////////////////////
  // Finally, compute E = chi1inv * D

  realnum *dmp[NUM_FIELD_COMPONENTS][2];
  FOR_FT_COMPONENTS(ft2, dc) DOCMP2 {
    dmp[dc][cmp] = f_minus_p[dc][cmp] ? f_minus_p[dc][cmp] : f[dc][cmp];
  }

  const std::vector<grid_volume> &update_volumes = gvs_eh[ft];
  const bool coalesce_cuda_tiles =
      cuda_session.active() && update_volumes.size() > 1 &&
      std::getenv("MEEP_GPU_DISABLE_TILE_COALESCING") == nullptr;
  const size_t execution_volume_count =
      coalesce_cuda_tiles ? 1 : update_volumes.size();
  bool recorded_tile_coalescing = false;
  for (size_t i = 0; i < execution_volume_count; ++i) {
    const grid_volume &update_volume =
        coalesce_cuda_tiles ? gv : update_volumes[i];
    DOCMP FOR_FT_COMPONENTS(ft, ec) {
      if (f[ec][cmp]) {
        if (type(ec) != ft) meep::abort("bug in FOR_FT_COMPONENTS");
        component dc = field_type_component(ft2, ec);
        const direction d_ec = component_direction(ec);
        const ptrdiff_t s_ec = gv.stride(d_ec) * (ft == H_stuff ? -1 : +1);
        const direction d_1 = cycle_direction(gv.dim, d_ec, 1);
        const component dc_1 = direction_component(dc, d_1);
        const ptrdiff_t s_1 = gv.stride(d_1) * (ft == H_stuff ? -1 : +1);
        const direction d_2 = cycle_direction(gv.dim, d_ec, 2);
        const component dc_2 = direction_component(dc, d_2);
        const ptrdiff_t s_2 = gv.stride(d_2) * (ft == H_stuff ? -1 : +1);

        direction dsigw0 = d_ec;
        direction dsigw = s->sigsize[dsigw0] > 1 ? dsigw0 : NO_DIRECTION;

        // lazily allocate any E/H fields that are needed (H==B initially)
        if (i == 0 && f[ec][cmp] == f[dc][cmp] &&
            (s->chi1inv[ec][d_ec] || have_f_minus_p || dsigw != NO_DIRECTION)) {
          f[ec][cmp] = allocate_copy(f[dc][cmp]);
          allocated_eh = true;
        }

        // lazily allocate W auxiliary field
        if (i == 0 && !f_w[ec][cmp] && dsigw != NO_DIRECTION) {
          f_w[ec][cmp] = allocate_copy(f[ec][cmp]);
          if (needs_W_notowned(ec)) allocated_eh = true; // communication needed
        }

        // for solve_cw, when W exists we get W and E from special variables
        if (f_w[ec][cmp] && skip_w_components) continue;

        // save W field from this timestep in f_w_prev if needed by pols
        if (i == 0 && needs_W_prev(ec)) {
          if (!f_w_prev[ec][cmp])
            f_w_prev[ec][cmp] =
                allocate_copy(f_w[ec][cmp] ? f_w[ec][cmp] : f[ec][cmp]);
          else {
#if MEEP_HAVE_CUDA
            if (cuda_session.active())
              gpu::detail::resident_copy_fp32(
                  cuda_session.cache(), f_w_prev[ec][cmp],
                  f_w[ec][cmp] ? f_w[ec][cmp] : f[ec][cmp], gv.ntot());
            else
#endif
            memcpy(f_w_prev[ec][cmp],
                   f_w[ec][cmp] ? f_w[ec][cmp] : f[ec][cmp],
                   sizeof(realnum) * gv.ntot());
          }
        }

        if (f[ec][cmp] != f[dc][cmp]) {
          const ivec update_is =
              update_volume.little_owned_corner0(ec);
          const ivec update_ie = update_volume.big_corner();
          const size_t update_points =
              update_point_count(update_is, update_ie);
          bool cuda_dispatched = false;

#if MEEP_HAVE_CUDA
          if (cuda_session.active()) {
            const gpu::detail::index_space_fp32 index_space =
                gpu::detail::make_index_space_fp32(
                    gv, update_is, update_ie, dsigw);
            const bool pml = dsigw != NO_DIRECTION;
            const gpu::detail::update_eh_material_fp32 material = {
                s->chi1inv[ec][d_ec],
                dmp[dc_1][cmp] ? s->chi1inv[ec][d_1] : nullptr,
                dmp[dc_2][cmp] ? s->chi1inv[ec][d_2] : nullptr,
                s->chi2[ec],
                s->chi3[ec],
                pml ? f_w[ec][cmp] : nullptr,
                pml ? s->sig[dsigw] : nullptr,
                pml ? s->kap[dsigw] : nullptr,
                pml ? static_cast<size_t>(s->sigsize[dsigw]) : 0};
            gpu::detail::resident_update_eh_fp32(
                cuda_session.cache(), f[ec][cmp], dmp[dc][cmp],
                dmp[dc_1][cmp], dmp[dc_2][cmp], gv.ntot(),
                index_space, s_ec, s_1, s_2, material);
            if (coalesce_cuda_tiles && !recorded_tile_coalescing) {
              gpu::detail::record_update_eh_tile_coalescing(
                  static_cast<std::uint64_t>(update_volumes.size()));
              recorded_tile_coalescing = true;
            }
            cuda_dispatched = true;
          }
#endif

          if (!cuda_dispatched) {
            STEP_UPDATE_EDHB(
                f[ec][cmp], ec, gv, update_is, update_ie, dmp[dc][cmp],
                dmp[dc_1][cmp], dmp[dc_2][cmp], s->chi1inv[ec][d_ec],
                dmp[dc_1][cmp] ? s->chi1inv[ec][d_1] : NULL,
                dmp[dc_2][cmp] ? s->chi1inv[ec][d_2] : NULL, s_ec, s_1,
                s_2, s->chi2[ec], s->chi3[ec], f_w[ec][cmp], dsigw,
                s->sig[dsigw], s->kap[dsigw]);
            gpu::detail::record_cpu_update_eh(update_points);
          }

          if (gv.dim == Dcyl) {
            ivec is = update_volume.little_owned_corner(ec);
            if (is.r() == 0) {
              ivec ie = update_volume.big_corner();
              ie.set_direction(R, 0);
              /* pass NULL for off-diagonal terms since they must be
                 zero at r=0 for an axisymmetric structure: */
#if MEEP_HAVE_CUDA
              if (cuda_session.active()) {
                const bool pml = dsigw != NO_DIRECTION;
                const gpu::detail::index_space_fp32 index_space =
                    gpu::detail::make_index_space_fp32(
                        gv, is, ie, dsigw);
                const gpu::detail::update_eh_material_fp32 material = {
                    s->chi1inv[ec][d_ec],
                    nullptr,
                    nullptr,
                    s->chi2[ec],
                    s->chi3[ec],
                    pml ? f_w[ec][cmp] : nullptr,
                    pml ? s->sig[dsigw] : nullptr,
                    pml ? s->kap[dsigw] : nullptr,
                    pml ? static_cast<size_t>(s->sigsize[dsigw]) : 0};
                gpu::detail::resident_update_eh_fp32(
                    cuda_session.cache(), f[ec][cmp], dmp[dc][cmp],
                    nullptr, nullptr, gv.ntot(), index_space, s_ec, s_1,
                    s_2, material);
              }
              else
#endif
              {
                STEP_UPDATE_EDHB(
                    f[ec][cmp], ec, gv, is, ie, dmp[dc][cmp], NULL, NULL,
                    s->chi1inv[ec][d_ec], NULL, NULL, s_ec, s_1, s_2,
                    s->chi2[ec], s->chi3[ec], f_w[ec][cmp], dsigw,
                    s->sig[dsigw], s->kap[dsigw]);
                gpu::detail::record_cpu_update_eh(
                    update_point_count(is, ie));
              }
            }
          }
        }
      }
    }
  }

  cuda_session.finish();
  return allocated_eh;
}

} // namespace meep
