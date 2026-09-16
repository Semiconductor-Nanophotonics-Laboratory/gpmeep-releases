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

#ifndef BICGSTAB_H
#define BICGSTAB_H

#include <stdexcept>
#include <string>

#include "meep.hpp"

namespace meep {

/* signature of a user callback function, passed to bicgstabL, that computes y = Ax.  data is
   an opaque pointer to any user data that you need, passed through from the Adata pointer to
   bicgstabL (e.g. this could be a data structure that records information about A, such as
   its size n). */
typedef void (*bicgstab_op)(const realnum *x, realnum *y, void *data);

// Vector backends use this exception only to report a collectively detected
// numerical breakdown to the common solver. bicgstabL converts it to the
// documented -1 status; configuration, allocation, and runtime failures
// remain ordinary exceptions.
class bicgstab_numerical_breakdown : public std::runtime_error {
public:
  explicit bicgstab_numerical_breakdown(const std::string &message)
      : std::runtime_error(message) {}
};

// Storage-policy interface for BiCGSTAB-L vectors.  The legacy CPU entry
// point below uses an internal host implementation; solve_cw may provide a
// CUDA-resident implementation while sharing the exact same solver control
// flow and scalar recurrence.
class bicgstab_vector_ops {
public:
  virtual ~bicgstab_vector_ops() {}
  virtual double dot(size_t n, const realnum *x,
                     const realnum *y) = 0;
  virtual double norm2(size_t n, const realnum *x) = 0;
  virtual void fill(size_t n, realnum *x, realnum value) = 0;
  virtual void copy(size_t n, realnum *destination,
                    const realnum *source) = 0;
  virtual void scale(size_t n, realnum *x, double value) = 0;
  virtual void xpay(size_t n, realnum *x, double value,
                    const realnum *y) = 0;
  // output = left - value * output.
  virtual void left_minus_scale(size_t n, realnum *output,
                                const realnum *left,
                                double value) = 0;
};

/* solve Ax=b by biCGSTAB(L).   L is the dimension of the subspace (typically < 10) — larger L
   gives more expensive iterations but faster convergence; n is the dimension of the vectors and
   matrices (x and b are length n, A is n by n); x is the initial solution guess on input (often
   zero) and the approximate solution on output; b is the right hand side; A is a function that
   implements A-times-vector (see above) and Adata is an opaque user data pointer passed through to
   A; tol is the stopping tolerance |Ax-b| < tol*|b|; on *iters is the maximum allowed number of
   iterations and on output it is the actual number; quiet=1 to suppress status messages during the
   solver.

   You must also allocate a workspace, usually by calling the routine twice: on the first call, pass
   NULL for the work pointer, and the return value nwork is the required size of the workspace. Then
   allocate work = new realnum[nwork], and call it again.

   For non-NULL nwork, returns 0 on success, 1 if the maximum number of iterations was reached, and
   -1 if a breakdown in convergence was detected. */
ptrdiff_t bicgstabL(const int L, const size_t n, realnum *x, bicgstab_op A, void *Adata,
                    const realnum *b, const double tol,
                    int *iters,    // input *iters = max iters, output = actual iters
                    realnum *work, // if you pass work=NULL, bicgstab returns nwork
                    const bool quiet);

ptrdiff_t bicgstabL_with_vector_ops(
    const int L, const size_t n, realnum *x, bicgstab_op A, void *Adata,
    const realnum *b, const double tol, int *iters, realnum *work,
    const bool quiet, bicgstab_vector_ops &vector_ops);

// Solve in bounded BiCGSTAB-L batches and recompute the true residual b-Ax
// between batches.  This limits recursive-residual drift in FP32 while
// preserving the original entry point and behavior unless explicitly used.
ptrdiff_t bicgstabL_restarted(
    const int L, const size_t n, realnum *x, bicgstab_op A, void *Adata,
    const realnum *b, const double tol, int *iters, realnum *work,
    const bool quiet, int restart_interval);
ptrdiff_t bicgstabL_restarted_with_vector_ops(
    const int L, const size_t n, realnum *x, bicgstab_op A, void *Adata,
    const realnum *b, const double tol, int *iters, realnum *work,
    const bool quiet, int restart_interval,
    bicgstab_vector_ops &vector_ops);

} // namespace meep

#endif /* BICGSTAB_H */
