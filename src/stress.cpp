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

/* Computation of the force spectrum via integration of the Maxwell
   stress tensor of the Fourier-transformed fields */

#include <meep.hpp>
#include "gpu_backend_internal.hpp"

#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace std;

namespace meep {

namespace {

bool stress_exception_is_in_flight() noexcept {
#if __cplusplus >= 201703L
  return std::uncaught_exceptions() > 0;
#else
  return std::uncaught_exception();
#endif
}

class distributed_force_reduction_failure_guard {
public:
  explicit distributed_force_reduction_failure_guard(bool armed)
      : armed_(armed) {}

  ~distributed_force_reduction_failure_guard() {
    if (armed_ && stress_exception_is_in_flight())
      meep::abort(
          "rank-local failure during a distributed DFT force reduction; "
          "aborting the communicator to prevent an MPI collective deadlock");
  }

  void dismiss() noexcept { armed_ = false; }

private:
  bool armed_;
};

void sum_force_to_all(const double *input, double *output,
                      size_t count) {
  size_t offset = 0;
  while (offset < count) {
    const int batch = static_cast<int>(std::min<size_t>(
        count - offset,
        static_cast<size_t>(std::numeric_limits<int>::max())));
    gpu::detail::record_dft_reduction_mpi_allreduce(
        static_cast<size_t>(batch) * sizeof(double));
    sum_to_all(input + offset, output + offset, batch);
    offset += static_cast<size_t>(batch);
  }
}

} // namespace

dft_force::dft_force(dft_chunk *offdiag1_, dft_chunk *offdiag2_, dft_chunk *diag_, double fmin,
                     double fmax, int Nf, const volume &where_)
    : where(where_) {
  freq = meep::linspace(fmin, fmax, Nf);
  offdiag1 = offdiag1_;
  offdiag2 = offdiag2_;
  diag = diag_;
  // where = new volume(where_.get_min_corner(), where_.get_max_corner());
}

dft_force::dft_force(dft_chunk *offdiag1_, dft_chunk *offdiag2_, dft_chunk *diag_,
                     const std::vector<double> &freq_, const volume &where_)
    : where(where_) {
  freq = freq_;
  offdiag1 = offdiag1_;
  offdiag2 = offdiag2_;
  diag = diag_;
  // where = new volume(where_.get_min_corner(), where_.get_max_corner());
}

dft_force::dft_force(dft_chunk *offdiag1_, dft_chunk *offdiag2_, dft_chunk *diag_,
                     const double *freq_, size_t Nfreq, const volume &where_)
    : freq(Nfreq), where(where_) {
  for (size_t i = 0; i < Nfreq; ++i)
    freq[i] = freq_[i];
  offdiag1 = offdiag1_;
  offdiag2 = offdiag2_;
  diag = diag_;
  // where = new volume(where_.get_min_corner(), where_.get_max_corner());
}

dft_force::dft_force(const dft_force &f) : where(f.where) {
  freq = f.freq;
  offdiag1 = f.offdiag1;
  offdiag2 = f.offdiag2;
  diag = f.diag;
  // where = new volume(f.where->get_min_corner(), f.where->get_max_corner());
}

dft_force::~dft_force() {
  gpu::detail::destroy_resident_dft_reduction_plans_for_owner(this);
}

void dft_force::remove() {
  gpu::detail::destroy_resident_dft_reduction_plans_for_owner(this);
  while (offdiag1) {
    dft_chunk *nxt = offdiag1->next_in_dft;
    delete offdiag1;
    offdiag1 = nxt;
  }
  while (offdiag2) {
    dft_chunk *nxt = offdiag2->next_in_dft;
    delete offdiag2;
    offdiag2 = nxt;
  }
  while (diag) {
    dft_chunk *nxt = diag->next_in_dft;
    delete diag;
    diag = nxt;
  }
}

void dft_force::operator-=(const dft_force &st) {
  if (offdiag1 && st.offdiag1) *offdiag1 -= *st.offdiag1;
  if (offdiag2 && st.offdiag2) *offdiag2 -= *st.offdiag2;
  if (diag && st.diag) *diag -= *st.diag;
}

static void stress_sum(size_t Nfreq, double *F, const dft_chunk *F1, const dft_chunk *F2) {
  for (const dft_chunk *cur = F1; cur; cur = cur->next_in_dft)
    gpu::detail::sync_resident_cache_for_owner(cur->fc);
  for (const dft_chunk *cur = F2; cur; cur = cur->next_in_dft)
    gpu::detail::sync_resident_cache_for_owner(cur->fc);
  for (const dft_chunk *curF1 = F1, *curF2 = F2; curF1 && curF2;
       curF1 = curF1->next_in_dft, curF2 = curF2->next_in_dft) {
    complex<double> extra_weight(real(curF1->extra_weight), imag(curF1->extra_weight));
    for (size_t k = 0; k < curF1->N; ++k)
      for (size_t i = 0; i < Nfreq; ++i)
        F[i] += real(extra_weight * complex<double>(curF1->dft[k * Nfreq + i]) *
                     conj(complex<double>(curF2->dft[k * Nfreq + i])));
  }
}

double *dft_force::force() {
  distributed_force_reduction_failure_guard distributed_failure_guard(
      count_processors() > 1);
  const size_t Nfreq = freq.size();
  const bool use_cuda = gpu::detail::cuda_active();
  std::vector<gpu::detail::dft_pair_reduction_request_fp32> requests;
  const gpu::detail::dft_pair_list_summary offdiag_summary =
      gpu::detail::append_dft_pair_reduction_requests(
          use_cuda ? &requests : nullptr, offdiag1, offdiag2,
          Nfreq, std::complex<double>(1.0, 0.0), true);
  const gpu::detail::dft_pair_list_summary diag_summary =
      gpu::detail::append_dft_pair_reduction_requests(
          use_cuda ? &requests : nullptr, diag, diag,
          Nfreq, std::complex<double>(1.0, 0.0), true);
  if (offdiag_summary.pair_count >
          std::numeric_limits<size_t>::max() -
              diag_summary.pair_count ||
      offdiag_summary.point_frequency_terms >
          std::numeric_limits<size_t>::max() -
              diag_summary.point_frequency_terms)
    throw std::overflow_error(
        "DFT force reduction work count overflow");
  const size_t pair_count =
      offdiag_summary.pair_count + diag_summary.pair_count;
  const size_t term_count =
      offdiag_summary.point_frequency_terms +
      diag_summary.point_frequency_terms;

  std::vector<double> local(Nfreq, 0.0);
  if (use_cuda && !requests.empty()) {
    if (Nfreq > std::numeric_limits<size_t>::max() / 2)
      throw std::overflow_error("DFT force result size overflow");
    std::vector<double> staged(2 * Nfreq, 0.0);
    gpu::detail::resident_reduce_dft_pairs_fp32(
        this, 0, requests.data(), requests.size(), Nfreq,
        staged.data());
    for (size_t i = 0; i < Nfreq; ++i)
      local[i] = staged[2 * i];
  }
  else if (!use_cuda) {
    stress_sum(Nfreq, local.data(), offdiag1, offdiag2);
    stress_sum(Nfreq, local.data(), diag, diag);
    gpu::detail::record_cpu_dft_reduction(pair_count, term_count);
  }

  std::unique_ptr<double[]> global(new double[Nfreq]);
  sum_force_to_all(local.data(), global.get(), Nfreq);
  distributed_failure_guard.dismiss();
  return global.release();
}

void dft_force::save_hdf5(h5file *file, const char *dprefix) {
  const dft_hdf5_dataset datasets[] = {
      {offdiag1, "offdiag1"}, {offdiag2, "offdiag2"}, {diag, "diag"}};
  save_dft_hdf5_many(datasets, 3, file, dprefix);
}

void dft_force::load_hdf5(h5file *file, const char *dprefix) {
  const dft_hdf5_dataset datasets[] = {
      {offdiag1, "offdiag1"}, {offdiag2, "offdiag2"}, {diag, "diag"}};
  load_dft_hdf5_many(datasets, 3, file, dprefix);
}

void dft_force::save_hdf5(fields &f, const char *fname, const char *dprefix, const char *prefix) {
  std::unique_ptr<h5file> ff(f.open_h5file(fname, h5file::WRITE, prefix));
  save_hdf5(ff.get(), dprefix);
}

void dft_force::load_hdf5(fields &f, const char *fname, const char *dprefix, const char *prefix) {
  std::unique_ptr<h5file> ff(f.open_h5file(fname, h5file::READONLY, prefix));
  load_hdf5(ff.get(), dprefix);
}

void dft_force::scale_dfts(complex<double> scale) {
  if (offdiag1) offdiag1->scale_dft(scale);
  if (offdiag2) offdiag2->scale_dft(scale);
  if (diag) diag->scale_dft(scale);
}

/* note that the components where->c indicate the direction of the
   force to be computed, so they should be vector components (such as
   Ex, Ey, ... or Sx, ...)  rather than pseudovectors (like Hx, ...). */
dft_force fields::add_dft_force(const volume_list *where, const double *freq, size_t Nfreq,
                                int decimation_factor) {
  dft_chunk *offdiag1 = 0, *offdiag2 = 0, *diag = 0;

  // copy where_ and add cc (conjugate-component) info for symmetry reduction
  volume_list where_copy(where);
  for (volume_list *w = &where_copy; w; w = w->next) {
    direction nd = normal_direction(w->v);
    if (nd == NO_DIRECTION) meep::abort("cannot determine dft_force normal");
    w->cc = direction_component(w->c, nd);
  }

  volume_list *where_reduced = S.reduce(&where_copy);
  if (!where_reduced) // empty list
    return dft_force(offdiag1, offdiag2, diag, freq, Nfreq, volume(v.center()));

  volume everywhere = where_reduced->v;

  for (volume_list *w = where_reduced; w; w = w->next) {
    direction nd = component_direction(w->cc); // normal direction
    direction fd = component_direction(w->c);  // force direction
    if (fd == NO_DIRECTION) meep::abort("NO_DIRECTION dft_force is invalid");
    if (coordinate_mismatch(gv.dim, fd)) meep::abort("coordinate-type mismatch in add_dft_force");

    if (fd != nd) { // off-diagonal stress-tensor terms
      offdiag1 = add_dft(direction_component(Ex, fd), w->v, freq, Nfreq, true, w->weight, offdiag1,
                         false, 1.0, true, 0, decimation_factor);
      offdiag2 = add_dft(direction_component(Ex, nd), w->v, freq, Nfreq, false, 1.0, offdiag2,
                         false, 1.0, true, 0, decimation_factor);
      offdiag1 = add_dft(direction_component(Hx, fd), w->v, freq, Nfreq, true, w->weight, offdiag1,
                         false, 1.0, true, 0, decimation_factor);
      offdiag2 = add_dft(direction_component(Hx, nd), w->v, freq, Nfreq, false, 1.0, offdiag2,
                         false, 1.0, true, 0, decimation_factor);
    }
    else // diagonal stress-tensor terms
      LOOP_OVER_FIELD_DIRECTIONS(gv.dim, d) {
        complex<double> weight1 = w->weight * (d == fd ? +0.5 : -0.5);
        diag = add_dft(direction_component(Ex, d), w->v, freq, Nfreq, true, 1.0, diag, true,
                       weight1, false, 0, decimation_factor);
        diag = add_dft(direction_component(Hx, d), w->v, freq, Nfreq, true, 1.0, diag, true,
                       weight1, false, 0, decimation_factor);
      }
    everywhere = everywhere | w->v;
  }

  delete where_reduced;
  return dft_force(offdiag1, offdiag2, diag, freq, Nfreq, everywhere);
}

} // namespace meep
