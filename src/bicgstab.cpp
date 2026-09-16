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

#include <math.h>
#include <string.h>

#include "meep/mympi.hpp"
#include "bicgstab.hpp"

#include "config.h"

/* bicgstab() implements an iterative solver for non-symmetric linear
   operators, using the algorithm described in:

      Gerard L. G. Sleijpen and Diederik R. Fokkema, "BiCGSTAB(L) for
      linear equations involving unsymmetric matrices with complex
      spectrum," Electronic Trans. on Numerical Analysis 1, 11-32
      (1993).

   and also:

      Gerard L.G. Sleijpen, Henk A. van der Vorst, and Diederik
      R. Fokkema, " BiCGstab(L) and other Hybrid Bi-CG Methods,"
      Numerical Algorithms 7, 75-109 (1994).

   This is a generalization of the stabilized biconjugate-gradient
   (BiCGSTAB) algorithm proposed by van der Vorst (and described
   in the book _Templates for the Solution of Linear Systems_ by
   Barrett et al.)  BiCGSTAB(1) is equivalent to BiCGSTAB, and
   BiCGSTAB(2) is a slightly more efficient version of the BiCGSTAB2
   algorithm by Gutknecht, while BiCGSTAB(L>2) is a further
   generalization.

   The reason that we use this generalization of BiCGSTAB is that the
   BiCGSTAB(1) algorithm was observed by Sleijpen and Fokkema to have
   poor (or even failing) convergence when the linear operator has
   near-pure imaginary eigenvalues.  This is precisely the case for
   our problem (the eigenvalues of the timestep operator are i*omega),
   and we observed precisely such stagnation of convergence.  The
   BiCGSTAB(2) algorithm was reported to fix most such convergence
   problems, and indeed L > 1 seems to converge well for us. */

/* Other variations to explore:

   G. L. G. Sleijpen and H. A. van der Vorst, "Reliable updated
   residuals in hybrid Bi-CG methods," Computing 56 (2), 141-163
   (1996).

   G. L. G. Sleijpen and H. A. van der Vorst, "Maintaining convergence
   properties of BiCGstab methods in finite precision arithmetic,"
   Numerical Algorithms 10, 203-223 (1995).

   See also code on Sleijpen's web page:
                 http://www.math.uu.nl/people/sleijpen/

*/

using namespace std;

namespace meep {

namespace {

class host_bicgstab_vector_ops : public bicgstab_vector_ops {
public:
  double dot(size_t n, const realnum *x,
             const realnum *y) override {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) sum += x[i] * y[i];
    const double global_sum = sum_to_all(sum);
    if (!std::isfinite(global_sum))
      throw bicgstab_numerical_breakdown(
          "nonfinite host Krylov dot product");
    return global_sum;
  }

  double norm2(size_t n, const realnum *x) override {
    // Do not implement this as sqrt(dot(x,x)): scaling avoids overflow.
    double xmax = 0;
    bool local_finite = true;
    for (size_t i = 0; i < n; ++i) {
      const double xabs = fabs(x[i]);
      local_finite = local_finite && std::isfinite(xabs);
      if (xabs > xmax) xmax = xabs;
    }
    const bool globally_finite = and_to_all(local_finite);
    xmax = max_to_all(xmax);
    if (!globally_finite || !std::isfinite(xmax))
      throw bicgstab_numerical_breakdown(
          "nonfinite host Krylov vector");
    if (xmax == 0) return 0;
    const double inverse = 1.0 / xmax;
    long double sum = 0;
    for (size_t i = 0; i < n; ++i) {
      const double scaled = inverse * x[i];
      sum += scaled * scaled;
    }
    const long double global_sum = sum_to_all(sum);
    const long double result = xmax * sqrt(global_sum);
    if (!std::isfinite(global_sum) || !std::isfinite(result))
      throw bicgstab_numerical_breakdown(
          "nonfinite host Krylov norm");
    return static_cast<double>(result);
  }

  void fill(size_t n, realnum *x, realnum value) override {
    for (size_t i = 0; i < n; ++i) x[i] = value;
  }

  void copy(size_t n, realnum *destination,
            const realnum *source) override {
    memcpy(destination, source, n * sizeof(realnum));
  }

  void scale(size_t n, realnum *x, double value) override {
    if (!std::isfinite(value))
      throw bicgstab_numerical_breakdown(
          "nonfinite host Krylov scale coefficient");
    for (size_t i = 0; i < n; ++i) x[i] *= value;
  }

  void xpay(size_t n, realnum *x, double value,
            const realnum *y) override {
    if (!std::isfinite(value))
      throw bicgstab_numerical_breakdown(
          "nonfinite host Krylov xpay coefficient");
    for (size_t i = 0; i < n; ++i) x[i] += value * y[i];
  }

  void left_minus_scale(size_t n, realnum *output,
                        const realnum *left,
                        double value) override {
    if (!std::isfinite(value))
      throw bicgstab_numerical_breakdown(
          "nonfinite host Krylov left-minus-scale coefficient");
    for (size_t i = 0; i < n; ++i)
      output[i] = left[i] - value * output[i];
  }
};

} // namespace

#define MEEP_MIN_OUTPUT_TIME 4.0 // output no more often than this many seconds

typedef realnum *prealnum; // grr, ISO C++ forbids new (double*)[...]

/* BiCGSTAB(L) algorithm for the n-by-n problem Ax = b */
ptrdiff_t bicgstabL_with_vector_ops(
    const int L, const size_t n, realnum *x, bicgstab_op A, void *Adata,
    const realnum *b, const double tol, int *iters, realnum *work,
    const bool quiet, bicgstab_vector_ops &vector_ops) {
  if (!work) return (2 * L + 3) * n; // required workspace

  prealnum *r = new prealnum[L + 1];
  prealnum *u = new prealnum[L + 1];
  for (int i = 0; i <= L; ++i) {
    r[i] = work + i * n;
    u[i] = work + (L + 1 + i) * n;
  }

  double bnrm = 1.0;
  int iter = 0;
  double last_output_wall_time = wall_time();

  double *gamma = new double[L + 1];
  double *gamma_p = new double[L + 1];
  double *gamma_pp = new double[L + 1];

  double *tau = new double[L * L];
  double *sigma = new double[L + 1];

  int ierr = 0; // error code to return, if any
  const double breaktol = 1e-30;

  /**** FIXME: check for breakdown conditions(?) during iteration  ****/

  std::string numerical_breakdown_message;
  try {
  bnrm = vector_ops.norm2(n, b);
  if (bnrm == 0.0) bnrm = 1.0;

  // rtilde = r[0] = b - Ax
  realnum *rtilde = work + (2 * L + 2) * n;
  A(x, r[0], Adata);
  vector_ops.left_minus_scale(n, r[0], b, 1.0);
  vector_ops.copy(n, rtilde, r[0]);

  { /* Sleipjen normalizes rtilde in his code; it seems to help slightly */
    const double rtilde_norm = vector_ops.norm2(n, rtilde);
    if (rtilde_norm != 0.0)
      vector_ops.scale(n, rtilde, 1.0 / rtilde_norm);
  }

  vector_ops.fill(n, u[0], 0); // u[0] = 0

  double rho = 1.0, alpha = 0, omega = 1;

  double resid;
  while ((resid = vector_ops.norm2(n, r[0])) > tol * bnrm) {
    ++iter;
    if (!quiet && wall_time() > last_output_wall_time + MEEP_MIN_OUTPUT_TIME) {
      master_printf("residual[%d] = %g\n", iter, resid / bnrm);
      last_output_wall_time = wall_time();
    }

    rho = -omega * rho;
    for (int j = 0; j < L; ++j) {
      if (fabs(rho) < breaktol) {
        ierr = -1;
        goto finish;
      }
      double rho1 = vector_ops.dot(n, r[j], rtilde);
      double beta = alpha * rho1 / rho;
      rho = rho1;
      for (int i = 0; i <= j; ++i)
        vector_ops.left_minus_scale(n, u[i], r[i], beta);
      A(u[j], u[j + 1], Adata);
      alpha = rho / vector_ops.dot(n, u[j + 1], rtilde);
      for (int i = 0; i <= j; ++i)
        vector_ops.xpay(n, r[i], -alpha, u[i + 1]);
      A(r[j], r[j + 1], Adata);
      vector_ops.xpay(n, x, alpha, u[0]);
    }

    for (int j = 1; j <= L; ++j) {
      for (int i = 1; i < j; ++i) {
        int ij = (j - 1) * L + (i - 1);
        tau[ij] = vector_ops.dot(n, r[j], r[i]) / sigma[i];
        vector_ops.xpay(n, r[j], -tau[ij], r[i]);
      }
      sigma[j] = vector_ops.dot(n, r[j], r[j]);
      gamma_p[j] = vector_ops.dot(n, r[0], r[j]) / sigma[j];
    }

    omega = gamma[L] = gamma_p[L];
    for (int j = L - 1; j >= 1; --j) {
      gamma[j] = gamma_p[j];
      for (int i = j + 1; i <= L; ++i)
        gamma[j] -= tau[(i - 1) * L + (j - 1)] * gamma[i];
    }
    for (int j = 1; j < L; ++j) {
      gamma_pp[j] = gamma[j + 1];
      for (int i = j + 1; i < L; ++i)
        gamma_pp[j] += tau[(i - 1) * L + (j - 1)] * gamma[i + 1];
    }

    vector_ops.xpay(n, x, gamma[1], r[0]);
    vector_ops.xpay(n, r[0], -gamma_p[L], r[L]);
    vector_ops.xpay(n, u[0], -gamma[L], u[L]);
    for (int j = 1; j < L; ++j) { /* TODO: use blas DGEMV (for L > 2) */
      vector_ops.xpay(n, x, gamma_pp[j], r[j]);
      vector_ops.xpay(n, r[0], -gamma_p[j], r[j]);
      vector_ops.xpay(n, u[0], -gamma[j], u[j]);
    }

    if (iter == *iters) {
      ierr = 1;
      break;
    }
  }

  if (!quiet)
    master_printf("final residual = %g\n",
                  vector_ops.norm2(n, r[0]) / bnrm);

  }
  catch (const bicgstab_numerical_breakdown &error) {
    ierr = -1;
    numerical_breakdown_message = error.what();
  }

finish:
  if (!quiet && ierr < 0)
    master_printf(
        "BiCGSTAB-L numerical breakdown%s%s\n",
        numerical_breakdown_message.empty() ? "" : ": ",
        numerical_breakdown_message.c_str());
  delete[] sigma;
  delete[] tau;
  delete[] gamma_pp;
  delete[] gamma_p;
  delete[] gamma;
  delete[] u;
  delete[] r;

  *iters = iter;
  return ierr;
}

ptrdiff_t bicgstabL(const int L, const size_t n, realnum *x,
                    bicgstab_op A, void *Adata, const realnum *b,
                    const double tol, int *iters, realnum *work,
                    const bool quiet) {
  if (!work) return (2 * L + 3) * n;
  host_bicgstab_vector_ops vector_ops;
  return bicgstabL_with_vector_ops(
      L, n, x, A, Adata, b, tol, iters, work, quiet, vector_ops);
}

ptrdiff_t bicgstabL_restarted_with_vector_ops(
    const int L, const size_t n, realnum *x, bicgstab_op A, void *Adata,
    const realnum *b, const double tol, int *iters, realnum *work,
    const bool quiet, int restart_interval,
    bicgstab_vector_ops &vector_ops) {
  if (!work) return (2 * L + 3) * n;
  if (!iters)
    throw std::invalid_argument(
        "restarted BiCGSTAB-L requires a non-null iteration limit");
  if (*iters < 0)
    throw std::invalid_argument(
        "restarted BiCGSTAB-L iteration limit must be nonnegative");
  if (restart_interval < 1)
    throw std::invalid_argument(
        "restarted BiCGSTAB-L interval must be positive");
  if (!std::isfinite(tol) || tol < 0.0)
    throw std::invalid_argument(
        "restarted BiCGSTAB-L tolerance must be finite and nonnegative");

  const int maximum_iterations = *iters;
  int completed_iterations = 0;
  ptrdiff_t status = 1;
  double rhs_norm = 1.0;
  try {
    rhs_norm = vector_ops.norm2(n, b);
  }
  catch (const bicgstab_numerical_breakdown &) {
    *iters = 0;
    return -1;
  }
  if (rhs_norm == 0.0) rhs_norm = 1.0;

  while (completed_iterations < maximum_iterations) {
    int batch_iterations = std::min(
        restart_interval, maximum_iterations - completed_iterations);
    status = bicgstabL_with_vector_ops(
        L, n, x, A, Adata, b, tol, &batch_iterations, work, true,
        vector_ops);
    completed_iterations += batch_iterations;
    if (status < 0) break;

    // The base algorithm tests a recursively updated residual.  Reuse the
    // first workspace vector for an independent true residual without
    // modifying x, even when the batch reported recursive convergence.
    double true_residual = 0.0;
    try {
      A(x, work, Adata);
      vector_ops.left_minus_scale(n, work, b, 1.0);
      true_residual = vector_ops.norm2(n, work);
    }
    catch (const bicgstab_numerical_breakdown &) {
      status = -1;
      break;
    }
    if (true_residual <= tol * rhs_norm) {
      status = 0;
      break;
    }
    if (batch_iterations == 0) {
      status = 1;
      break;
    }
    status = 1;
  }

  if (!quiet && status >= 0) {
    try {
      A(x, work, Adata);
      vector_ops.left_minus_scale(n, work, b, 1.0);
      master_printf("final true residual = %g\n",
                    vector_ops.norm2(n, work) / rhs_norm);
    }
    catch (const bicgstab_numerical_breakdown &) {
      status = -1;
    }
  }
  *iters = completed_iterations;
  return status;
}

ptrdiff_t bicgstabL_restarted(
    const int L, const size_t n, realnum *x, bicgstab_op A, void *Adata,
    const realnum *b, const double tol, int *iters, realnum *work,
    const bool quiet, int restart_interval) {
  if (!work) return (2 * L + 3) * n;
  host_bicgstab_vector_ops vector_ops;
  return bicgstabL_restarted_with_vector_ops(
      L, n, x, A, Adata, b, tol, iters, work, quiet,
      restart_interval, vector_ops);
}

} // namespace meep
