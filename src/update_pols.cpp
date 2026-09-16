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
#include <math.h>
#include <string.h>
#include <assert.h>

#include "meep.hpp"
#include "meep_internals.hpp"
#include "gpu_backend_internal.hpp"
#include "config.h"

#include <stdexcept>
#include <typeinfo>

using namespace std;

namespace meep {

void fields::update_pols(field_type ft) {
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine())
      if (chunks[i]->update_pols(ft)) {
        chunk_connections_valid = false;
        assert(changed_materials);
      }
}

bool fields_chunk::update_pols(field_type ft) {
  bool allocated_fields = false;

  realnum *w[NUM_FIELD_COMPONENTS][2];
  FOR_COMPONENTS(c) DOCMP2 { w[c][cmp] = f_w[c][cmp] ? f_w[c][cmp] : f[c][cmp]; }

  const bool cuda_backend_active = gpu::detail::cuda_active();
  const bool cuda_phase_eligible =
      cuda_backend_active && sizeof(realnum) == sizeof(float);
  if (cuda_backend_active)
    for (polarization_state *p = pol[ft]; p; p = p->next) {
      const bool standard_lorentzian =
          p->s &&
          typeid(*p->s) == typeid(lorentzian_susceptibility);
      const bool standard_gyrotropic =
          p->s &&
          typeid(*p->s) == typeid(gyrotropic_susceptibility);
      const bool standard_multilevel =
          p->s &&
          typeid(*p->s) == typeid(multilevel_susceptibility);
      if (!standard_lorentzian && !standard_gyrotropic &&
          !standard_multilevel)
        throw std::runtime_error(
            "required CUDA polarization update supports only standard "
            "Lorentzian/Drude, gyrotropic, and multilevel "
            "susceptibilities");
      if (standard_gyrotropic) {
        const gyrotropic_susceptibility *gyrotropic =
            static_cast<const gyrotropic_susceptibility *>(p->s);
        std::string reason;
        if (!gyrotropic->cuda_update_P_eligible(w, gv, &reason))
          throw std::runtime_error(
              "required CUDA gyrotropic polarization is unsupported: " +
              reason);
      }
      if (standard_multilevel) {
        const multilevel_susceptibility *multilevel =
            static_cast<const multilevel_susceptibility *>(p->s);
        std::string reason;
        if (!multilevel->cuda_update_P_eligible(
                w, f_w_prev, gv, &reason))
          throw std::runtime_error(
              "required CUDA multilevel polarization is unsupported: " +
              reason);
      }
    }
  gpu::detail::resident_curl_session cuda_session(this,
                                                   cuda_phase_eligible);

  for (polarization_state *p = pol[ft]; p; p = p->next) {

    // Lazily allocate internal polarization data:
    if (!p->data) {
      p->data = p->s->new_internal_data(f, gv);
      if (p->data) {
        p->s->init_internal_data(f, dt, gv, p->data);
        allocated_fields = true;
      }
    }

    // Finally, timestep the polarizations:
    bool cuda_dispatched = false;
#if MEEP_HAVE_CUDA
    const lorentzian_susceptibility *lorentzian =
        dynamic_cast<const lorentzian_susceptibility *>(p->s);
    const bool standard_lorentzian =
        lorentzian &&
        typeid(*p->s) == typeid(lorentzian_susceptibility);
    if (cuda_session.active() && standard_lorentzian)
      cuda_dispatched = lorentzian->update_P_cuda(
          cuda_session.cache(), w, f_w_prev, dt, gv, p->data);
    const gyrotropic_susceptibility *gyrotropic =
        dynamic_cast<const gyrotropic_susceptibility *>(p->s);
    const bool standard_gyrotropic =
        gyrotropic &&
        typeid(*p->s) == typeid(gyrotropic_susceptibility);
    if (cuda_session.active() && standard_gyrotropic)
      cuda_dispatched = gyrotropic->update_P_cuda(
          cuda_session.cache(), w, f_w_prev, dt, gv, p->data);
    const multilevel_susceptibility *multilevel =
        dynamic_cast<const multilevel_susceptibility *>(p->s);
    const bool standard_multilevel =
        multilevel &&
        typeid(*p->s) == typeid(multilevel_susceptibility);
    if (cuda_session.active() && standard_multilevel)
      cuda_dispatched = multilevel->update_P_cuda(
          cuda_session.cache(), w, f_w_prev, dt, gv, p->data);
#endif
    if (!cuda_dispatched) {
      if (cuda_backend_active)
        throw std::runtime_error(
            "preflighted CUDA polarization update was not "
            "dispatched");
      p->s->update_P(w, f_w_prev, dt, gv, p->data);
      size_t points = 0;
      FOR_COMPONENTS(c) DOCMP2 {
        if (p->s->needs_P(c, cmp, f)) points += gv.nowned(c);
      }
      gpu::detail::record_cpu_polarization(points);
    }
  }

  cuda_session.finish();
  return allocated_fields;
}

} // namespace meep
