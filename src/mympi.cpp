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

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <stdarg.h>
#include <string.h>
#include <utility>
#include <vector>

#include "meep.hpp"
#include "config.h"

#ifdef HAVE_MPI
#include <mpi.h>
#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#endif
#endif

#ifdef _OPENMP
#include "omp.h"
#else
#define omp_get_num_threads() (1)
#endif

#ifdef IGNORE_SIGFPE
#include <signal.h>
#endif

#if defined(DEBUG) && defined(HAVE_FEENABLEEXCEPT)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <fenv.h>
#if !HAVE_DECL_FEENABLEEXCEPT
extern "C" int feenableexcept(int EXCEPTS);
#endif
#endif

#if HAVE_SYS_TIME_H
#include <sys/time.h>
#include <time.h>
#else
#include <time.h>
#endif
#ifdef HAVE_BSDGETTIMEOFDAY
#ifndef HAVE_GETTIMEOFDAY
#define gettimeofday BSDgettimeofday
#define HAVE_GETTIMEOFDAY 1
#endif
#endif

#if HAVE_IMMINTRIN_H
#include <immintrin.h>
#endif

#define UNUSED(x) (void)x // silence compiler warnings

#define MPI_REALNUM (sizeof(realnum) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT)

using namespace std;

namespace meep {

namespace {

std::atomic<std::uint64_t> eager_cuda_receive_start_calls(0);
std::atomic<std::uint64_t> eager_cuda_send_start_calls(0);
std::atomic<std::uint64_t> eager_cuda_receive_requests(0);
std::atomic<std::uint64_t> eager_cuda_send_requests(0);

#ifdef HAVE_MPI
MPI_Comm mycomm = MPI_COMM_WORLD;
int cached_node_rank = 0;
int cached_node_size = 1;
bool cached_node_topology = false;
int validated_cuda_device_transport = -1;
int validated_mpi_completion_policy = -1;
// MPI_TAG_UB is guaranteed to be at least 32767. Some MPI implementations do
// not copy predefined communicator attributes onto split communicators, so
// cache the world value during initialization and use it conservatively for
// every Meep subcommunicator.
int cached_mpi_tag_upper_bound = 32767;
#if MPI_VERSION >= 3 && MEEP_HAVE_CUDA
struct gpu_assignment_claim {
  char identifier[80];
  int allow_duplicates;
};
MPI_Win gpu_assignment_claim_window = MPI_WIN_NULL;
gpu_assignment_claim *gpu_assignment_claim_storage = nullptr;
int gpu_assignment_claim_count = 0;
#endif

bool mpi_runtime_ready() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) return false;
  int finalized = 0;
  MPI_Finalized(&finalized);
  return !finalized;
}

int launcher_topology_value(const char *const *names, std::size_t count,
                            int fallback) {
  for (std::size_t index = 0; index < count; ++index) {
    const char *value = std::getenv(names[index]);
    if (!value || !*value) continue;
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end && *end == '\0' && parsed >= 0 && parsed <= INT_MAX)
      return static_cast<int>(parsed);
  }
  return fallback;
}

void cache_node_topology() {
  if (!mpi_runtime_ready() || cached_node_topology) return;
#if MPI_VERSION >= 3
  MPI_Comm node_comm = MPI_COMM_NULL;
  int global_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  const int split_error =
      MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, global_rank,
                          MPI_INFO_NULL, &node_comm);
  if (split_error != MPI_SUCCESS || node_comm == MPI_COMM_NULL)
    meep::abort(
        "MPI_Comm_split_type failed while discovering the node-local topology");
  const int rank_error = MPI_Comm_rank(node_comm, &cached_node_rank);
  const int size_error = MPI_Comm_size(node_comm, &cached_node_size);
  MPI_Comm_free(&node_comm);
  if (rank_error != MPI_SUCCESS || size_error != MPI_SUCCESS)
    meep::abort(
        "MPI_Comm_rank/size failed while discovering the node-local topology");
#else
  int global_rank = 0;
  int global_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &global_size);
  const char *rank_names[] = {"OMPI_COMM_WORLD_LOCAL_RANK",
                              "MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID",
                              "PMI_LOCAL_RANK"};
  const char *size_names[] = {"OMPI_COMM_WORLD_LOCAL_SIZE",
                              "MV2_COMM_WORLD_LOCAL_SIZE", "SLURM_NTASKS_PER_NODE",
                              "PMI_LOCAL_SIZE"};
  cached_node_rank = launcher_topology_value(
      rank_names, sizeof(rank_names) / sizeof(rank_names[0]), global_rank);
  cached_node_size = launcher_topology_value(
      size_names, sizeof(size_names) / sizeof(size_names[0]), global_size);
#endif
  cached_node_topology = true;
}

void cache_mpi_tag_upper_bound() {
  int *tag_upper_bound = nullptr;
  int present = 0;
  if (MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_upper_bound,
                        &present) == MPI_SUCCESS &&
      present && tag_upper_bound && *tag_upper_bound >= 32767)
    cached_mpi_tag_upper_bound = *tag_upper_bound;
}

void create_gpu_assignment_claim_window() {
#if MPI_VERSION >= 3 && MEEP_HAVE_CUDA
  if (gpu_assignment_claim_window != MPI_WIN_NULL) return;
  int world_rank = 0;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &gpu_assignment_claim_count) !=
          MPI_SUCCESS)
    meep::abort("MPI world query failed while creating GPU claim table");
  const MPI_Aint bytes =
      world_rank == 0
          ? static_cast<MPI_Aint>(
                sizeof(gpu_assignment_claim) *
                static_cast<std::size_t>(gpu_assignment_claim_count))
          : 0;
  void *storage = nullptr;
  if (MPI_Win_allocate(bytes, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &storage,
                       &gpu_assignment_claim_window) != MPI_SUCCESS)
    meep::abort("MPI GPU claim-table allocation failed");
  if (world_rank == 0) {
    gpu_assignment_claim_storage =
        static_cast<gpu_assignment_claim *>(storage);
    std::memset(gpu_assignment_claim_storage, 0,
                static_cast<std::size_t>(bytes));
    if (MPI_Win_sync(gpu_assignment_claim_window) != MPI_SUCCESS)
      meep::abort("MPI GPU claim-table initialization failed");
  }
  if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS)
    meep::abort("MPI GPU claim-table initialization barrier failed");
#endif
}

void destroy_gpu_assignment_claim_window() {
#if MPI_VERSION >= 3 && MEEP_HAVE_CUDA
  if (mpi_runtime_ready() && gpu_assignment_claim_window != MPI_WIN_NULL &&
      MPI_Win_free(&gpu_assignment_claim_window) != MPI_SUCCESS)
    meep::abort("MPI GPU claim-table destruction failed");
  gpu_assignment_claim_storage = nullptr;
  gpu_assignment_claim_count = 0;
#endif
}

bool mpi_cuda_support_is_reported() {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
#if defined(OMPI_MAJOR_VERSION)
  // MPIX_Query_cuda_support reports whether this Open MPI installation was
  // built with CUDA support. Open MPI can still disable that support at
  // runtime (opal_cuda_support=false), in which case handing MPI a device
  // pointer can make a host-only BTL dereference it. Query the active MCA
  // control variable through MPI_T as well.
  if (!mpi_runtime_ready()) return false;
  static int cached_runtime_support = -1;
  if (cached_runtime_support >= 0) return cached_runtime_support != 0;
  if (MPIX_Query_cuda_support() == 0) {
    cached_runtime_support = 0;
    return false;
  }
#if MPI_VERSION >= 3
  int provided = MPI_THREAD_SINGLE;
  if (MPI_T_init_thread(MPI_THREAD_SINGLE, &provided) != MPI_SUCCESS) {
    cached_runtime_support = 0;
    return false;
  }

  bool enabled = false;
  int cvar_index = -1;
  if (MPI_T_cvar_get_index("opal_cuda_support", &cvar_index) == MPI_SUCCESS) {
    char name[64] = {};
    int name_length = sizeof(name);
    int verbosity = 0;
    MPI_Datatype datatype = MPI_DATATYPE_NULL;
    MPI_T_enum enumeration = MPI_T_ENUM_NULL;
    char description[1] = {};
    int description_length = 0;
    int binding = 0;
    int scope = 0;
    if (MPI_T_cvar_get_info(
            cvar_index, name, &name_length, &verbosity, &datatype,
            &enumeration, description, &description_length, &binding,
            &scope) == MPI_SUCCESS) {
      MPI_T_cvar_handle handle = MPI_T_CVAR_HANDLE_NULL;
      int value_count = 0;
      if (MPI_T_cvar_handle_alloc(
              cvar_index, nullptr, &handle, &value_count) == MPI_SUCCESS) {
        union {
          int integer;
          unsigned char bytes[sizeof(int)];
        } value = {};
        if (value_count == 1 &&
            MPI_T_cvar_read(handle, &value) == MPI_SUCCESS)
          enabled =
              datatype == MPI_INT ? value.integer != 0 : value.bytes[0] != 0;
        MPI_T_cvar_handle_free(&handle);
      }
    }
  }
  MPI_T_finalize();
  cached_runtime_support = enabled ? 1 : 0;
  return enabled;
#else
  cached_runtime_support = 0;
  return false;
#endif
#else
  return true;
#endif
#else
  return false;
#endif
}

int parse_cuda_device_transport_request() {
  const char *value = std::getenv("MEEP_GPU_MPI_TRANSPORT");
  if (!value || !*value || strcmp(value, "auto") == 0)
    return 1;
  if (strcmp(value, "pinned") == 0 ||
      strcmp(value, "host") == 0)
    return 0;
  if (strcmp(value, "cuda-aware") == 0 ||
      strcmp(value, "device") == 0)
    return 2;
  throw std::invalid_argument(
      std::string("invalid MEEP_GPU_MPI_TRANSPORT='") + value +
      "' (expected auto, pinned, or cuda-aware)");
}

int parse_mpi_completion_policy_request() {
  const char *value = std::getenv("MEEP_GPU_MPI_COMPLETION");
  if (!value || !*value || strcmp(value, "waitsome") == 0)
    return 0;
  if (strcmp(value, "waitall") == 0)
    return 1;
  throw std::invalid_argument(
      std::string("invalid MEEP_GPU_MPI_COMPLETION='") + value +
      "' (expected waitsome or waitall)");
}
#endif

// comms_manager implementation that uses MPI.
bool exception_is_in_flight() noexcept {
#if __cplusplus >= 201703L
  return std::uncaught_exceptions() > 0;
#else
  return std::uncaught_exception();
#endif
}

class mpi_comms_manager : public comms_manager {
public:
  mpi_comms_manager()
#ifdef HAVE_MPI
      : cuda_device_buffers_(validated_cuda_device_transport == 1),
        waitall_completion_(validated_mpi_completion_policy == 1),
        cuda_request_plan_prepared_(false),
        cuda_receives_posted_(false), cuda_sends_posted_(false)
#endif
  {}
  ~mpi_comms_manager() noexcept override {
#ifdef HAVE_MPI
    if (!has_pending_work()) return;
    // Never post queued host transfers or execute user callbacks while the
    // stack is already unwinding: their captured state may have been
    // destroyed before this manager.  A distributed partial exchange cannot
    // be recovered locally, so fail immediately instead of hanging in wait.
    if (exception_is_in_flight())
      meep::abort(
          "communications manager destroyed during exceptional MPI boundary "
          "exchange");
#endif
    try {
      finish();
    }
    catch (const std::exception &error) {
      meep::abort(
          "communications manager cleanup failed: %s", error.what());
    }
    catch (...) {
      meep::abort("communications manager cleanup failed");
    }
  }

  void finish() {
#ifdef HAVE_MPI
    if (!has_pending_work()) return;
    finishing_guard finish_in_progress(finishing_);
    if (cuda_device_buffers_) {
      if (!cuda_request_plan_prepared_)
        prepare_cuda_device_request_plan();
      if (!cuda_receives_posted_)
        post_cuda_device_receives(false);
      if (!cuda_sends_posted_)
        post_cuda_device_sends(false);
    }
    else
      post_aggregated_host_requests();
    if (reqs.size() != callbacks.size())
      meep::abort("MPI boundary request/callback state is inconsistent");
    if (reqs.size() >
        static_cast<size_t>(std::numeric_limits<int>::max()))
      meep::abort("too many pending MPI boundary requests");
    std::exception_ptr callback_error;
    if (cuda_device_buffers_ && waitall_completion_) {
      if (!reqs.empty() &&
          MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(),
                      MPI_STATUSES_IGNORE) != MPI_SUCCESS)
        meep::abort("MPI_Waitall failed during boundary exchange");
      for (std::size_t request_idx = 0; request_idx < callbacks.size();
           ++request_idx)
        try {
          callbacks[request_idx]();
        }
        catch (...) {
          if (!callback_error) callback_error = std::current_exception();
        }
    }
    else {
      int num_pending_requests = static_cast<int>(reqs.size());
      completed_indices_.resize(static_cast<size_t>(num_pending_requests));
      while (num_pending_requests) {
        int num_completed_requests = 0;
        const int wait_error =
            MPI_Waitsome(static_cast<int>(reqs.size()), reqs.data(),
                         &num_completed_requests, completed_indices_.data(),
                         MPI_STATUSES_IGNORE);
        if (wait_error != MPI_SUCCESS ||
            num_completed_requests == MPI_UNDEFINED)
          meep::abort("MPI_Waitsome failed during boundary exchange");
        for (int i = 0; i < num_completed_requests; ++i) {
          int request_idx = completed_indices_[static_cast<size_t>(i)];
          try {
            callbacks[request_idx]();
          }
          catch (...) {
            if (!callback_error) callback_error = std::current_exception();
          }
          reqs[request_idx] = MPI_REQUEST_NULL;
          --num_pending_requests;
        }
      }
    }
    for (MPI_Datatype &datatype : datatypes)
      if (datatype != MPI_DATATYPE_NULL &&
          MPI_Type_free(&datatype) != MPI_SUCCESS)
        meep::abort("MPI_Type_free failed during boundary exchange");
    reqs.clear();
    callbacks.clear();
    datatypes.clear();
    pending_receives.clear();
    pending_sends.clear();
    cuda_request_plan_prepared_ = false;
    cuda_receives_posted_ = false;
    cuda_sends_posted_ = false;
    if (callback_error) std::rethrow_exception(callback_error);
#endif
  }

  void start_cuda_device_receives() {
#ifdef HAVE_MPI
    if (finishing_)
      throw std::logic_error(
          "cannot start CUDA-aware receives while communications manager is finishing");
    if (!cuda_device_buffers_)
      throw std::logic_error(
          "communications manager does not use CUDA device buffers");
    if (cuda_receives_posted_)
      throw std::logic_error("CUDA-aware receives were already posted");
    if (!cuda_request_plan_prepared_)
      prepare_cuda_device_request_plan();
    post_cuda_device_receives(true);
#else
    throw std::runtime_error(
        "CUDA-aware MPI receives require an MPI build");
#endif
  }

  void start_cuda_device_sends() {
#ifdef HAVE_MPI
    if (finishing_)
      throw std::logic_error(
          "cannot start CUDA-aware sends while communications manager is finishing");
    if (!cuda_device_buffers_)
      throw std::logic_error(
          "communications manager does not use CUDA device buffers");
    if (!cuda_request_plan_prepared_ || !cuda_receives_posted_)
      throw std::logic_error(
          "CUDA-aware receives must be posted before sends");
    if (cuda_sends_posted_)
      throw std::logic_error("CUDA-aware sends were already posted");
    post_cuda_device_sends(true);
#else
    throw std::runtime_error(
        "CUDA-aware MPI sends require an MPI build");
#endif
  }

  void send_real_async(const void *buf, size_t count, int dest, int tag) override {
#ifdef HAVE_MPI
    if (finishing_)
      throw std::logic_error(
          "cannot enqueue while communications manager is finishing");
    if (cuda_request_plan_prepared_)
      throw std::logic_error(
          "cannot enqueue after CUDA-aware request posting has started");
    if (cuda_device_buffers_ &&
        count > static_cast<size_t>(std::numeric_limits<int>::max()))
      meep::abort("MPI CUDA-aware boundary send is too large");
    // Queue CUDA-aware transfers too. finish() preallocates every request,
    // callback, and completion index before posting the first MPI operation,
    // eliminating allocation failures with active device-buffer requests.
    pending_sends.push_back(
        {const_cast<void *>(buf), count, dest, tag, [] {}});
#else
    (void)buf;
    (void)count;
    (void)dest;
    (void)tag;
#endif
  }

  void receive_real_async(void *buf, size_t count, int source, int tag,
                          const receive_callback &cb) override {
#ifdef HAVE_MPI
    if (finishing_)
      throw std::logic_error(
          "cannot enqueue while communications manager is finishing");
    if (cuda_request_plan_prepared_)
      throw std::logic_error(
          "cannot enqueue after CUDA-aware request posting has started");
    if (cuda_device_buffers_ &&
        count > static_cast<size_t>(std::numeric_limits<int>::max()))
      meep::abort("MPI CUDA-aware boundary receive is too large");
    pending_receives.push_back({buf, count, source, tag, cb});
#else
    (void)buf;
    (void)count;
    (void)source;
    (void)tag;
    (void)cb;
#endif
  }

#ifdef HAVE_MPI
  size_t max_transfer_size() const override { return std::numeric_limits<int>::max(); }
  bool supports_cuda_device_buffers() const {
    return cuda_device_buffers_;
  }
  bool uses_waitall_completion() const {
    return cuda_device_buffers_ && waitall_completion_;
  }
  size_t physical_message_count() const {
    // Once finish() has committed and posted a plan, reqs is the exact
    // physical-message set. Do not also count the still-owned logical
    // transfers while completion callbacks are being drained.
    if (!reqs.empty()) return reqs.size();
    if (cuda_device_buffers_)
      return pending_receives.size() + pending_sends.size();
    std::map<int, bool> receive_peers;
    std::map<int, bool> send_peers;
    for (const pending_transfer &transfer : pending_receives)
      receive_peers[transfer.peer] = true;
    for (const pending_transfer &transfer : pending_sends)
      send_peers[transfer.peer] = true;
    return receive_peers.size() + send_peers.size();
  }
#endif

private:
#ifdef HAVE_MPI
  class finishing_guard {
  public:
    explicit finishing_guard(bool &finishing) : finishing_(finishing) {
      if (finishing_)
        throw std::logic_error(
            "communications manager finish is not reentrant");
      finishing_ = true;
    }
    ~finishing_guard() { finishing_ = false; }

  private:
    bool &finishing_;
  };

  struct pending_transfer {
    void *buffer;
    size_t count;
    int peer;
    int tag;
    receive_callback callback;
  };

  struct planned_host_request {
    bool receive = false;
    int peer = 0;
    int tag = 0;
    void *buffer = nullptr;
    int count = 0;
    std::vector<int> block_lengths;
    std::vector<MPI_Aint> displacements;
    MPI_Datatype datatype = MPI_DATATYPE_NULL;
    receive_callback callback;
  };

  void prepare_cuda_device_request_plan() {
    if (cuda_request_plan_prepared_)
      throw std::logic_error(
          "CUDA-aware MPI request plan was already prepared");
    if (!reqs.empty() || !callbacks.empty())
      meep::abort(
          "CUDA-aware MPI request planning started from nonempty state");
    const std::size_t request_count =
        pending_receives.size() + pending_sends.size();
    if (request_count >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
      meep::abort("too many CUDA-aware MPI boundary requests");

    // Build the complete request/callback plan in local storage.  A throwing
    // std::function copy therefore leaves the manager unchanged and reusable;
    // only no-throw vector swaps commit the plan before the first MPI post.
    std::vector<receive_callback> planned_callbacks;
    std::vector<MPI_Request> planned_requests(
        request_count, MPI_REQUEST_NULL);
    std::vector<int> planned_completed_indices(request_count);
    planned_callbacks.reserve(request_count);
    for (const pending_transfer &transfer : pending_receives)
      planned_callbacks.push_back(transfer.callback);
    for (std::size_t index = 0; index < pending_sends.size(); ++index)
      planned_callbacks.push_back([] {});
    callbacks.swap(planned_callbacks);
    reqs.swap(planned_requests);
    completed_indices_.swap(planned_completed_indices);
    cuda_request_plan_prepared_ = true;
  }

  void post_cuda_device_receives(bool eager) {
    if (!cuda_request_plan_prepared_ || cuda_receives_posted_)
      throw std::logic_error(
          "CUDA-aware MPI receive posting state is invalid");
    std::size_t request_index = 0;
    for (const pending_transfer &transfer : pending_receives) {
      if (MPI_Irecv(transfer.buffer, static_cast<int>(transfer.count),
                    MPI_REALNUM, transfer.peer, transfer.tag, mycomm,
                    &reqs[request_index]) != MPI_SUCCESS)
        meep::abort(
            "MPI_Irecv failed during CUDA-aware boundary exchange");
      ++request_index;
    }
    cuda_receives_posted_ = true;
    if (eager && !pending_receives.empty()) {
      eager_cuda_receive_start_calls.fetch_add(1);
      eager_cuda_receive_requests.fetch_add(
          static_cast<std::uint64_t>(pending_receives.size()));
    }
  }

  void post_cuda_device_sends(bool eager) {
    if (!cuda_request_plan_prepared_ || !cuda_receives_posted_ ||
        cuda_sends_posted_)
      throw std::logic_error(
          "CUDA-aware MPI send posting state is invalid");
    std::size_t request_index = pending_receives.size();
    for (const pending_transfer &transfer : pending_sends) {
      if (MPI_Isend(transfer.buffer, static_cast<int>(transfer.count),
                    MPI_REALNUM, transfer.peer, transfer.tag, mycomm,
                    &reqs[request_index]) != MPI_SUCCESS)
        meep::abort(
            "MPI_Isend failed during CUDA-aware boundary exchange");
      ++request_index;
    }
    cuda_sends_posted_ = true;
    if (eager && !pending_sends.empty()) {
      eager_cuda_send_start_calls.fetch_add(1);
      eager_cuda_send_requests.fetch_add(
          static_cast<std::uint64_t>(pending_sends.size()));
    }
  }

  static void append_aggregated_host_plan(
      std::vector<pending_transfer *> group, bool receive,
      std::vector<planned_host_request> &plans) {
    if (group.empty()) return;
    std::sort(
        group.begin(), group.end(),
        [](const pending_transfer *left, const pending_transfer *right) {
          return left->tag < right->tag;
        });

    planned_host_request plan;
    plan.receive = receive;
    plan.peer = group.front()->peer;
    plan.tag = group.front()->tag;
    if (group.size() == 1) {
      pending_transfer *transfer = group.front();
      if (transfer->count >
          static_cast<size_t>(std::numeric_limits<int>::max()))
        meep::abort("MPI boundary block is too large");
      plan.buffer = transfer->buffer;
      plan.count = static_cast<int>(transfer->count);
      plan.callback = transfer->callback;
      plans.push_back(std::move(plan));
      return;
    }

    if (group.size() >
        static_cast<size_t>(std::numeric_limits<int>::max()))
      meep::abort("too many MPI boundary blocks to aggregate");
    plan.block_lengths.resize(group.size());
    plan.displacements.resize(group.size());
    for (size_t index = 0; index < group.size(); ++index) {
      if (group[index]->count >
          static_cast<size_t>(std::numeric_limits<int>::max()))
        meep::abort("MPI boundary block is too large to aggregate");
      plan.block_lengths[index] = static_cast<int>(group[index]->count);
      if (MPI_Get_address(group[index]->buffer,
                          &plan.displacements[index]) != MPI_SUCCESS)
        meep::abort("MPI_Get_address failed for a boundary block");
    }
    plan.buffer = MPI_BOTTOM;
    plan.count = 1;
    plan.callback = [group] {
      std::exception_ptr callback_error;
      for (pending_transfer *transfer : group)
        try {
          transfer->callback();
        }
        catch (...) {
          if (!callback_error) callback_error = std::current_exception();
        }
      if (callback_error) std::rethrow_exception(callback_error);
    };
    plans.push_back(std::move(plan));
  }

  void post_aggregated_host_requests() {
    std::map<int, std::vector<pending_transfer *> > receives_by_peer;
    std::map<int, std::vector<pending_transfer *> > sends_by_peer;
    for (pending_transfer &transfer : pending_receives)
      receives_by_peer[transfer.peer].push_back(&transfer);
    for (pending_transfer &transfer : pending_sends)
      sends_by_peer[transfer.peer].push_back(&transfer);
    std::vector<planned_host_request> plans;
    plans.reserve(receives_by_peer.size() + sends_by_peer.size());
    for (auto &entry : receives_by_peer)
      append_aggregated_host_plan(
          std::move(entry.second), true, plans);
    for (auto &entry : sends_by_peer)
      append_aggregated_host_plan(
          std::move(entry.second), false, plans);

    // All C++ allocations and callback copies finish before the first
    // nonblocking request is posted. From this point through posting, only
    // no-throw moves into pre-reserved storage are performed.
    reqs.reserve(reqs.size() + plans.size());
    callbacks.reserve(callbacks.size() + plans.size());
    datatypes.reserve(datatypes.size() + plans.size());
    completed_indices_.resize(reqs.size() + plans.size());

    for (planned_host_request &plan : plans)
      if (!plan.block_lengths.empty()) {
        if (MPI_Type_create_hindexed(
                static_cast<int>(plan.block_lengths.size()),
                plan.block_lengths.data(), plan.displacements.data(),
                MPI_REALNUM, &plan.datatype) != MPI_SUCCESS ||
            MPI_Type_commit(&plan.datatype) != MPI_SUCCESS)
          meep::abort(
              "failed to create an aggregated MPI boundary datatype");
      }

    for (planned_host_request &plan : plans) {
      callbacks.push_back(std::move(plan.callback));
      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Datatype wire_type = MPI_REALNUM;
      if (plan.datatype != MPI_DATATYPE_NULL) {
        wire_type = plan.datatype;
        datatypes.push_back(plan.datatype);
        plan.datatype = MPI_DATATYPE_NULL;
      }
      const int post_error =
          plan.receive
              ? MPI_Irecv(plan.buffer, plan.count, wire_type,
                          plan.peer, plan.tag, mycomm, &reqs.back())
              : MPI_Isend(plan.buffer, plan.count, wire_type,
                          plan.peer, plan.tag, mycomm, &reqs.back());
      if (post_error != MPI_SUCCESS)
        meep::abort("failed to post an aggregated MPI boundary request");
    }
  }

  bool has_pending_work() const {
    return !reqs.empty() || !pending_receives.empty() ||
           !pending_sends.empty() || cuda_request_plan_prepared_;
  }

  std::vector<MPI_Request> reqs;
  std::vector<MPI_Datatype> datatypes;
  std::vector<int> completed_indices_;
  std::vector<pending_transfer> pending_receives;
  std::vector<pending_transfer> pending_sends;
  bool cuda_device_buffers_;
  bool waitall_completion_;
  bool cuda_request_plan_prepared_;
  bool cuda_receives_posted_;
  bool cuda_sends_posted_;
#endif
  bool finishing_ = false;
  std::vector<receive_callback> callbacks;
};

} // namespace

void initialize_distributed_gpu_runtime() {
#ifdef HAVE_MPI
  if (!mpi_runtime_ready())
    meep::abort(
        "distributed GPU runtime initialization requires initialized MPI");
  // Both operations are collective over MPI_COMM_WORLD.  Python invokes this
  // immediately after mpi4py initialization, before Meep can split its active
  // communicator for parallel objective functions.
  cache_node_topology();
  cache_mpi_tag_upper_bound();
  create_gpu_assignment_claim_window();
#endif
}

void finalize_distributed_gpu_runtime() {
#ifdef HAVE_MPI
  release_distributed_device_identifier();
  destroy_gpu_assignment_claim_window();
#endif
}

std::unique_ptr<comms_manager> create_comms_manager() {
  return std::unique_ptr<comms_manager>(new mpi_comms_manager());
}

bool comms_supports_cuda_device_buffers(const comms_manager *manager) {
#ifdef HAVE_MPI
  const mpi_comms_manager *mpi_manager =
      dynamic_cast<const mpi_comms_manager *>(manager);
  return mpi_manager && mpi_manager->supports_cuda_device_buffers();
#else
  (void)manager;
  return false;
#endif
}

bool comms_uses_waitall_completion(const comms_manager *manager) {
#ifdef HAVE_MPI
  const mpi_comms_manager *mpi_manager =
      dynamic_cast<const mpi_comms_manager *>(manager);
  return mpi_manager && mpi_manager->uses_waitall_completion();
#else
  (void)manager;
  return false;
#endif
}

void comms_start_cuda_device_receives(comms_manager *manager) {
  mpi_comms_manager *mpi_manager =
      dynamic_cast<mpi_comms_manager *>(manager);
  if (!mpi_manager)
    throw std::invalid_argument(
        "Meep communications manager implementation is unavailable");
  mpi_manager->start_cuda_device_receives();
}

void comms_start_cuda_device_sends(comms_manager *manager) {
  mpi_comms_manager *mpi_manager =
      dynamic_cast<mpi_comms_manager *>(manager);
  if (!mpi_manager)
    throw std::invalid_argument(
        "Meep communications manager implementation is unavailable");
  mpi_manager->start_cuda_device_sends();
}

void reset_comms_overlap_statistics() noexcept {
  eager_cuda_receive_start_calls.store(0);
  eager_cuda_send_start_calls.store(0);
  eager_cuda_receive_requests.store(0);
  eager_cuda_send_requests.store(0);
}

comms_overlap_statistics get_comms_overlap_statistics() noexcept {
  return {eager_cuda_receive_start_calls.load(),
          eager_cuda_send_start_calls.load(),
          eager_cuda_receive_requests.load(),
          eager_cuda_send_requests.load()};
}

size_t comms_physical_message_count(const comms_manager *manager) {
#ifdef HAVE_MPI
  const mpi_comms_manager *mpi_manager =
      dynamic_cast<const mpi_comms_manager *>(manager);
  return mpi_manager ? mpi_manager->physical_message_count() : 0;
#else
  (void)manager;
  return 0;
#endif
}

void comms_finish(comms_manager *manager) {
  mpi_comms_manager *mpi_manager =
      dynamic_cast<mpi_comms_manager *>(manager);
  if (!mpi_manager)
    throw std::invalid_argument(
        "Meep communications manager implementation is unavailable");
  mpi_manager->finish();
}

int verbosity = 1; // defined in meep.h

/* Set CPU to flush subnormal values to zero (if iszero == true).  This slightly
   reduces the range of floating-point numbers, but can greatly increase the speed
   in cases where subnormal values might arise (e.g. deep in the tails of
   exponentially decaying sources).

   See also meep#1708.

   code based on github.com/JuliaLang/julia/blob/master/src/processor_x86.cpp#L1087-L1104,
   which is free software under the GPL-compatible "MIT license" */
static void _set_zero_subnormals(bool iszero) {
#if HAVE_IMMINTRIN_H
  unsigned int flags =
      0x00008040; // assume a non-ancient processor with SSE2, supporting both FTZ and DAZ flags
  unsigned int state = _mm_getcsr();
  if (iszero)
    state |= flags;
  else
    state &= ~flags;
  _mm_setcsr(state);
#else
  (void)iszero; // unused
#endif
}
void set_zero_subnormals(bool iszero) {
#ifdef _OPENMP
#pragma omp parallel
  { _set_zero_subnormals(iszero); }
#else
  _set_zero_subnormals(iszero);
#endif
}

void setup() {
#ifdef _OPENMP
  if (getenv("OMP_NUM_THREADS") == NULL) omp_set_num_threads(1);
#endif
  set_zero_subnormals(true);
}

initialize::initialize(int &argc, char **&argv) {
#ifdef HAVE_MPI
#ifdef _OPENMP
  // setup() applies the same default after MPI initialization, but the
  // requested MPI thread level must be checked against the effective Meep
  // team size rather than the OpenMP runtime's host-wide default.
  if (getenv("OMP_NUM_THREADS") == NULL) omp_set_num_threads(1);
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED && omp_get_max_threads() > 1)
    abort("MPI does not support multi-threaded execution");
#else
  MPI_Init(&argc, &argv);
#endif
  // MPI_Comm_split_type is collective. Cache the physical node topology once
  // while every rank is known to be inside Meep initialization; GPU backend
  // queries can then be safely made by only a subset of ranks.
  initialize_distributed_gpu_runtime();
  int major, minor;
  MPI_Get_version(&major, &minor);
  if (verbosity > 0)
    master_printf("Using MPI version %d.%d, %d processes\n", major, minor, count_processors());
#else
  UNUSED(argc);
  UNUSED(argv);
#endif
#if defined(DEBUG_FP) && defined(HAVE_FEENABLEEXCEPT)
  feenableexcept(FE_INVALID | FE_OVERFLOW); // crash if NaN created, or overflow
#endif
#ifdef IGNORE_SIGFPE
  signal(SIGFPE, SIG_IGN);
#endif
  t_start = wall_time();
  setup();
}

initialize::~initialize() {
  if (verbosity > 0) master_printf("\nElapsed run time = %g s\n", elapsed_time());
#ifdef HAVE_MPI
  end_divide_parallel();
  finalize_distributed_gpu_runtime();
  MPI_Finalize();
#endif
}

double wall_time(void) {
#ifdef HAVE_MPI
  return MPI_Wtime();
#elif defined(_OPENMP)
  return omp_get_wtime();
#elif HAVE_GETTIMEOFDAY
  struct timeval tv;
  gettimeofday(&tv, 0);
  return (tv.tv_sec + tv.tv_usec * 1e-6);
#else
  return (clock() * 1.0 / CLOCKS_PER_SECOND);
#endif
}

[[noreturn]] void abort(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  char *s;
  vasprintf(&s, fmt, ap);
  va_end(ap);
  // Make a std::string to support older compilers (std::runtime_error(char *) was added in C++11)
  std::string error_msg(s);
  free(s);
#ifdef HAVE_MPI
  if (count_processors() == 1) { throw runtime_error("meep: " + error_msg); }
  fprintf(stderr, "meep: %s", error_msg.c_str());
  if (fmt[strlen(fmt) - 1] != '\n') fputc('\n', stderr); // force newline
  MPI_Abort(MPI_COMM_WORLD, 1);
  std::abort(); // Unreachable but MPI_Abort does not have the noreturn attribute.
#else
  throw runtime_error("meep: " + error_msg);
#endif
}

void send(int from, int to, double *data, int size) {
#ifdef HAVE_MPI
  if (from == to) return;
  if (size == 0) return;
  const int me = my_rank();
  if (from == me) MPI_Send(data, size, MPI_DOUBLE, to, 1, mycomm);
  MPI_Status stat;
  if (to == me) MPI_Recv(data, size, MPI_DOUBLE, from, 1, mycomm, &stat);
#else
  UNUSED(from);
  UNUSED(to);
  UNUSED(data);
  UNUSED(size);
#endif
}

void broadcast(int from, float *data, int size) {
#ifdef HAVE_MPI
  if (size == 0) return;
  MPI_Bcast(data, size, MPI_FLOAT, from, mycomm);
#else
  UNUSED(from);
  UNUSED(data);
  UNUSED(size);
#endif
}

void broadcast(int from, double *data, int size) {
#ifdef HAVE_MPI
  if (size == 0) return;
  MPI_Bcast(data, size, MPI_DOUBLE, from, mycomm);
#else
  UNUSED(from);
  UNUSED(data);
  UNUSED(size);
#endif
}

void broadcast(int from, char *data, int size) {
#ifdef HAVE_MPI
  if (size == 0) return;
  MPI_Bcast(data, size, MPI_CHAR, from, mycomm);
#else
  UNUSED(from);
  UNUSED(data);
  UNUSED(size);
#endif
}

void broadcast(int from, complex<double> *data, int size) {
#ifdef HAVE_MPI
  if (size == 0) return;
  MPI_Bcast(data, 2 * size, MPI_DOUBLE, from, mycomm);
#else
  UNUSED(from);
  UNUSED(data);
  UNUSED(size);
#endif
}

void broadcast(int from, int *data, int size) {
#ifdef HAVE_MPI
  if (size == 0) return;
  MPI_Bcast(data, size, MPI_INT, from, mycomm);
#else
  UNUSED(from);
  UNUSED(data);
  UNUSED(size);
#endif
}

void broadcast(int from, size_t *data, int size) {
#ifdef HAVE_MPI
  if (size == 0) return;
  MPI_Bcast(data, size, sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG_LONG, from, mycomm);
#else
  UNUSED(from);
  UNUSED(data);
  UNUSED(size);
#endif
}

complex<double> broadcast(int from, complex<double> data) {
#ifdef HAVE_MPI
  MPI_Bcast(&data, 2, MPI_DOUBLE, from, mycomm);
#else
  UNUSED(from);
#endif
  return data;
}

double broadcast(int from, double data) {
#ifdef HAVE_MPI
  MPI_Bcast(&data, 1, MPI_DOUBLE, from, mycomm);
#else
  UNUSED(from);
#endif
  return data;
}

int broadcast(int from, int data) {
#ifdef HAVE_MPI
  MPI_Bcast(&data, 1, MPI_INT, from, mycomm);
#else
  UNUSED(from);
#endif
  return data;
}

bool broadcast(int from, bool b) { return broadcast(from, (int)b); }

double max_to_master(double in) {
  double out = in;
#ifdef HAVE_MPI
  MPI_Reduce(&in, &out, 1, MPI_DOUBLE, MPI_MAX, 0, mycomm);
#endif
  return out;
}

double max_to_all(double in) {
  double out = in;
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE, MPI_MAX, mycomm);
#endif
  return out;
}

int max_to_all(int in) {
  int out = in;
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MAX, mycomm);
#endif
  return out;
}

void max_to_all(const int *in, int *out, int size) {
#ifdef HAVE_MPI
  MPI_Allreduce((void *)in, out, size, MPI_INT, MPI_MAX, mycomm);
#else
  memcpy(out, in, sizeof(int) * size);
#endif
}

int min_to_all(int in) {
  int out = in;
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, mycomm);
#endif
  return out;
}

ivec max_to_all(const ivec &pt) {
  int in[5], out[5];
  for (int i = 0; i < 5; ++i)
    in[i] = out[i] = pt.in_direction(direction(i));
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 5, MPI_INT, MPI_MAX, mycomm);
#endif
  ivec ptout(pt.dim);
  for (int i = 0; i < 5; ++i)
    ptout.set_direction(direction(i), out[i]);
  return ptout;
}

float sum_to_master(float in) {
  float out = in;
#ifdef HAVE_MPI
  MPI_Reduce(&in, &out, 1, MPI_FLOAT, MPI_SUM, 0, mycomm);
#endif
  return out;
}

double sum_to_master(double in) {
  double out = in;
#ifdef HAVE_MPI
  MPI_Reduce(&in, &out, 1, MPI_DOUBLE, MPI_SUM, 0, mycomm);
#endif
  return out;
}

double sum_to_all(double in) {
  double out = in;
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE, MPI_SUM, mycomm);
#endif
  return out;
}

void sum_to_all(const float *in, float *out, int size) {
#ifdef HAVE_MPI
  MPI_Allreduce((void *)in, out, size, MPI_FLOAT, MPI_SUM, mycomm);
#else
  memcpy(out, in, sizeof(float) * size);
#endif
}

void sum_to_all(const double *in, double *out, int size) {
#ifdef HAVE_MPI
  MPI_Allreduce((void *)in, out, size, MPI_DOUBLE, MPI_SUM, mycomm);
#else
  memcpy(out, in, sizeof(double) * size);
#endif
}

void sum_to_master(const float *in, float *out, int size) {
#ifdef HAVE_MPI
  MPI_Reduce((void *)in, out, size, MPI_FLOAT, MPI_SUM, 0, mycomm);
#else
  memcpy(out, in, sizeof(float) * size);
#endif
}

void sum_to_master(const double *in, double *out, int size) {
#ifdef HAVE_MPI
  MPI_Reduce((void *)in, out, size, MPI_DOUBLE, MPI_SUM, 0, mycomm);
#else
  memcpy(out, in, sizeof(double) * size);
#endif
}

void sum_to_all(const float *in, double *out, int size) {
  double *in2 = new double[size];
  for (int i = 0; i < size; ++i)
    in2[i] = in[i];
  sum_to_all(in2, out, size);
  delete[] in2;
}

void sum_to_all(const complex<double> *in, complex<double> *out, int size) {
  sum_to_all((const double *)in, (double *)out, 2 * size);
}

void sum_to_all(const complex<float> *in, complex<double> *out, int size) {
  sum_to_all((const float *)in, (double *)out, 2 * size);
}

void sum_to_all(const complex<float> *in, complex<float> *out, int size) {
  sum_to_all((const float *)in, (float *)out, 2 * size);
}

void sum_to_master(const complex<float> *in, complex<float> *out, int size) {
  sum_to_master((const float *)in, (float *)out, 2 * size);
}

void sum_to_master(const complex<double> *in, complex<double> *out, int size) {
  sum_to_master((const double *)in, (double *)out, 2 * size);
}

long double sum_to_all(long double in) {
  long double out = in;
#ifdef HAVE_MPI
  if (MPI_LONG_DOUBLE == MPI_DATATYPE_NULL)
    out = sum_to_all(double(in));
  else
    MPI_Allreduce(&in, &out, 1, MPI_LONG_DOUBLE, MPI_SUM, mycomm);
#endif
  return out;
}

int sum_to_all(int in) {
  int out = in;
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_SUM, mycomm);
#endif
  return out;
}

int partial_sum_to_all(int in) {
  int out = in;
#ifdef HAVE_MPI
  MPI_Scan(&in, &out, 1, MPI_INT, MPI_SUM, mycomm);
#endif
  return out;
}

size_t sum_to_all(size_t in) {
  size_t out = in;
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 1, sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                mycomm);
#endif
  return out;
}

void sum_to_all(const size_t *in, size_t *out, int size) {
#ifdef HAVE_MPI
  MPI_Allreduce((void *)in, out, size, sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG_LONG,
                MPI_SUM, mycomm);
#else
  memcpy(out, in, sizeof(size_t) * size);
#endif
}

void sum_to_master(const size_t *in, size_t *out, int size) {
#ifdef HAVE_MPI
  MPI_Reduce((void *)in, out, size, sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG_LONG,
             MPI_SUM, 0, mycomm);
#else
  memcpy(out, in, sizeof(size_t) * size);
#endif
}

size_t partial_sum_to_all(size_t in) {
  size_t out = in;
#ifdef HAVE_MPI
  MPI_Scan(&in, &out, 1, sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG_LONG, MPI_SUM,
           mycomm);
#endif
  return out;
}

complex<double> sum_to_all(complex<double> in) {
  complex<double> out = in;
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 2, MPI_DOUBLE, MPI_SUM, mycomm);
#endif
  return out;
}

complex<long double> sum_to_all(complex<long double> in) {
  complex<long double> out = in;
#ifdef HAVE_MPI
  if (MPI_LONG_DOUBLE == MPI_DATATYPE_NULL) {
    complex<double> dout;
    dout = sum_to_all(complex<double>(double(in.real()), double(in.imag())));
    out = complex<long double>(dout.real(), dout.imag());
  }
  else
    MPI_Allreduce(&in, &out, 2, MPI_LONG_DOUBLE, MPI_SUM, mycomm);
#endif
  return out;
}

bool or_to_all(bool in) {
  int in2 = in, out;
#ifdef HAVE_MPI
  MPI_Allreduce(&in2, &out, 1, MPI_INT, MPI_LOR, mycomm);
#else
  out = in2;
#endif
  return (bool)out;
}

void or_to_all(const int *in, int *out, int size) {
#ifdef HAVE_MPI
  MPI_Allreduce((void *)in, out, size, MPI_INT, MPI_LOR, mycomm);
#else
  memcpy(out, in, sizeof(int) * size);
#endif
}

void bw_or_to_all(const size_t *in, size_t *out, int size) {
#ifdef HAVE_MPI
  MPI_Allreduce((void *)in, out, size, sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG_LONG,
                MPI_BOR, mycomm);
#else
  memcpy(out, in, sizeof(size_t) * size);
#endif
}

bool and_to_all(bool in) {
  int in2 = in, out;
#ifdef HAVE_MPI
  MPI_Allreduce(&in2, &out, 1, MPI_INT, MPI_LAND, mycomm);
#else
  out = in2;
#endif
  return (bool)out;
}

void and_to_all(const int *in, int *out, int size) {
#ifdef HAVE_MPI
  MPI_Allreduce((void *)in, out, size, MPI_INT, MPI_LAND, mycomm);
#else
  memcpy(out, in, sizeof(int) * size);
#endif
}

void all_wait() {
#ifdef HAVE_MPI
  MPI_Barrier(mycomm);
#endif
}

int my_rank() {
#ifdef HAVE_MPI
  if (!mpi_runtime_ready()) return 0;
  int rank;
  MPI_Comm_rank(mycomm, &rank);
  return rank;
#else
  return 0;
#endif
}

int count_processors() {
#ifdef HAVE_MPI
  if (!mpi_runtime_ready()) return 1;
  int n;
  MPI_Comm_size(mycomm, &n);
  return n;
#else
  return 1;
#endif
}

int my_node_rank() {
#ifdef HAVE_MPI
  if (!mpi_runtime_ready()) return 0;
  if (cached_node_topology) return cached_node_rank;
  const char *names[] = {"OMPI_COMM_WORLD_LOCAL_RANK",
                         "MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID",
                         "PMI_LOCAL_RANK"};
  return launcher_topology_value(
      names, sizeof(names) / sizeof(names[0]), my_global_rank());
#else
  return 0;
#endif
}

int count_node_processors() {
#ifdef HAVE_MPI
  if (!mpi_runtime_ready()) return 1;
  if (cached_node_topology) return cached_node_size;
  const char *names[] = {"OMPI_COMM_WORLD_LOCAL_SIZE",
                         "MV2_COMM_WORLD_LOCAL_SIZE", "SLURM_NTASKS_PER_NODE",
                         "PMI_LOCAL_SIZE"};
  return launcher_topology_value(
      names, sizeof(names) / sizeof(names[0]), count_processors());
#else
  return 1;
#endif
}

bool distributed_device_identifiers_are_unique(
    const char *identifier, bool allow_duplicates) {
#ifdef HAVE_MPI
  if (!mpi_runtime_ready()) return identifier && *identifier;
  struct assignment {
    char identifier[80];
    int allow_duplicates;
  };
  assignment local = {};
  if (identifier && *identifier) {
    const std::size_t length = std::strlen(identifier);
    if (length >= sizeof(local.identifier))
      meep::abort("CUDA device identifier is too long");
    std::memcpy(local.identifier, identifier, length);
  }
  local.allow_duplicates = allow_duplicates ? 1 : 0;

  const int communicator_size = count_processors();
  std::vector<assignment> gathered(
      static_cast<std::size_t>(communicator_size));
  if (MPI_Allgather(&local, sizeof(local), MPI_BYTE, gathered.data(),
                    sizeof(local), MPI_BYTE, mycomm) != MPI_SUCCESS)
    meep::abort("MPI_Allgather failed while validating GPU assignments");

  for (int left = 0; left < communicator_size; ++left) {
    if (!gathered[static_cast<std::size_t>(left)].identifier[0])
      return false;
    for (int right = left + 1; right < communicator_size; ++right) {
      const assignment &left_assignment =
          gathered[static_cast<std::size_t>(left)];
      const assignment &right_assignment =
          gathered[static_cast<std::size_t>(right)];
      if (std::strncmp(left_assignment.identifier,
                       right_assignment.identifier,
                       sizeof(left_assignment.identifier)) == 0 &&
          !(left_assignment.allow_duplicates &&
            right_assignment.allow_duplicates))
        return false;
    }
  }
  return true;
#else
  (void)allow_duplicates;
  return identifier && *identifier;
#endif
}

bool claim_distributed_device_identifier(
    const char *identifier, bool allow_duplicates) {
#if defined(HAVE_MPI) && MPI_VERSION >= 3 && MEEP_HAVE_CUDA
  if (!mpi_runtime_ready()) return identifier && *identifier;
  if (!identifier || !*identifier ||
      gpu_assignment_claim_window == MPI_WIN_NULL)
    return false;
  const std::size_t identifier_length = std::strlen(identifier);
  if (identifier_length >= sizeof(gpu_assignment_claim::identifier))
    return false;

  int world_rank = 0;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) != MPI_SUCCESS)
    meep::abort("MPI world-rank query failed while claiming a GPU");
  std::vector<gpu_assignment_claim> claims(
      static_cast<std::size_t>(gpu_assignment_claim_count));
  gpu_assignment_claim desired = {};
  std::memcpy(desired.identifier, identifier, identifier_length);
  desired.allow_duplicates = allow_duplicates ? 1 : 0;

  if (MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0,
                   gpu_assignment_claim_window) != MPI_SUCCESS)
    meep::abort("MPI GPU claim-table lock failed");
  const int table_bytes = static_cast<int>(
      sizeof(gpu_assignment_claim) *
      static_cast<std::size_t>(gpu_assignment_claim_count));
  if (MPI_Get(claims.data(), table_bytes, MPI_BYTE, 0, 0, table_bytes,
              MPI_BYTE, gpu_assignment_claim_window) != MPI_SUCCESS ||
      MPI_Win_flush(0, gpu_assignment_claim_window) != MPI_SUCCESS)
    meep::abort("MPI GPU claim-table read failed");

  bool available = true;
  for (int rank = 0; rank < gpu_assignment_claim_count; ++rank) {
    if (rank == world_rank) continue;
    const gpu_assignment_claim &claim =
        claims[static_cast<std::size_t>(rank)];
    if (std::strncmp(claim.identifier, desired.identifier,
                     sizeof(claim.identifier)) == 0 &&
        claim.identifier[0] &&
        !(claim.allow_duplicates && desired.allow_duplicates)) {
      available = false;
      break;
    }
  }
  gpu_assignment_claim stored = available ? desired : gpu_assignment_claim{};
  const MPI_Aint displacement =
      static_cast<MPI_Aint>(
          sizeof(gpu_assignment_claim) *
          static_cast<std::size_t>(world_rank));
  if (MPI_Put(&stored, sizeof(stored), MPI_BYTE, 0, displacement,
              sizeof(stored), MPI_BYTE,
              gpu_assignment_claim_window) != MPI_SUCCESS ||
      MPI_Win_unlock(0, gpu_assignment_claim_window) != MPI_SUCCESS)
    meep::abort("MPI GPU claim-table update failed");
  return available;
#else
  (void)allow_duplicates;
  return identifier && *identifier;
#endif
}

void release_distributed_device_identifier() {
#if defined(HAVE_MPI) && MPI_VERSION >= 3 && MEEP_HAVE_CUDA
  if (!mpi_runtime_ready() ||
      gpu_assignment_claim_window == MPI_WIN_NULL)
    return;
  int world_rank = 0;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) != MPI_SUCCESS)
    meep::abort("MPI world-rank query failed while releasing a GPU");
  const gpu_assignment_claim empty = {};
  const MPI_Aint displacement =
      static_cast<MPI_Aint>(
          sizeof(gpu_assignment_claim) *
          static_cast<std::size_t>(world_rank));
  if (MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0,
                   gpu_assignment_claim_window) != MPI_SUCCESS ||
      MPI_Put(&empty, sizeof(empty), MPI_BYTE, 0, displacement,
              sizeof(empty), MPI_BYTE,
              gpu_assignment_claim_window) != MPI_SUCCESS ||
      MPI_Win_unlock(0, gpu_assignment_claim_window) != MPI_SUCCESS)
    meep::abort("MPI GPU claim-table release failed");
#endif
}

void validate_distributed_mpi_transport() {
#ifdef HAVE_MPI
  int local_request = -1;
  int local_completion_policy = -1;
  std::string local_transport_error;
  std::string local_completion_error;
  try {
    local_request = parse_cuda_device_transport_request();
  }
  catch (const std::exception &error) {
    local_transport_error = error.what();
  }
  try {
    local_completion_policy = parse_mpi_completion_policy_request();
  }
  catch (const std::exception &error) {
    local_completion_error = error.what();
  }
  const int minimum_request = min_to_all(local_request);
  const int maximum_request = max_to_all(local_request);
  const int minimum_completion_policy = min_to_all(local_completion_policy);
  const int maximum_completion_policy = max_to_all(local_completion_policy);
  const bool every_rank_cuda_aware =
      and_to_all(mpi_cuda_support_is_reported());
  const bool pinned_and_device_conflict =
      minimum_request == 0 && maximum_request == 2;
  const bool device_unavailable =
      maximum_request == 2 && !every_rank_cuda_aware;
  if (minimum_request < 0 || pinned_and_device_conflict ||
      device_unavailable) {
    validated_cuda_device_transport = 0;
    std::ostringstream message;
    message << "distributed CUDA ranks selected incompatible MPI transports";
    if (device_unavailable)
      message << ": CUDA-aware transport was required but is not active on "
                 "every rank";
    if (!local_transport_error.empty())
      message << ": " << local_transport_error;
    throw std::runtime_error(message.str());
  }
  // Automatic ranks negotiate the safe intersection: pinned if any rank
  // explicitly requests it or lacks CUDA-aware support, otherwise device.
  validated_cuda_device_transport =
      maximum_request == 2 ||
              (minimum_request == 1 && every_rank_cuda_aware)
          ? 1
          : 0;
  if (minimum_completion_policy < 0 ||
      minimum_completion_policy != maximum_completion_policy) {
    validated_mpi_completion_policy = 0;
    std::ostringstream message;
    message << "distributed CUDA ranks selected incompatible MPI completion "
               "policies";
    if (!local_completion_error.empty())
      message << ": " << local_completion_error;
    throw std::runtime_error(message.str());
  }
  validated_mpi_completion_policy = maximum_completion_policy;
#endif
}

int max_communication_tag() {
#ifdef HAVE_MPI
  if (!mpi_runtime_ready()) return INT_MAX;
  return cached_mpi_tag_upper_bound;
#else
  return INT_MAX;
#endif
}

bool with_mpi() {
#ifdef HAVE_MPI
  return true;
#else
  return false;
#endif
}

// IO Routines...

bool am_really_master() { return (my_global_rank() == 0); }

static meep_printf_callback_func master_printf_callback = NULL;
static meep_printf_callback_func master_printf_stderr_callback = NULL;

meep_printf_callback_func set_meep_printf_callback(meep_printf_callback_func func) {
  meep_printf_callback_func old_func = master_printf_callback;
  master_printf_callback = func;
  return old_func;
}

meep_printf_callback_func set_meep_printf_stderr_callback(meep_printf_callback_func func) {
  meep_printf_callback_func old_func = master_printf_stderr_callback;
  master_printf_stderr_callback = func;
  return old_func;
}

static void _do_master_printf(FILE *output, meep_printf_callback_func callback, const char *fmt,
                              va_list ap) {
  if (am_really_master()) {
    if (callback) {
      char *s;
      vasprintf(&s, fmt, ap);
      callback(s);
      free(s);
    }
    else {
      vfprintf(output, fmt, ap);
      fflush(output);
    }
  }
  va_end(ap);
}

void master_printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  _do_master_printf(stdout, master_printf_callback, fmt, ap);
  va_end(ap);
}

void master_printf_stderr(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  _do_master_printf(stderr, master_printf_stderr_callback, fmt, ap);
  va_end(ap);
}

static FILE *debf = NULL;

void debug_printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  if (debf == NULL) {
    char temp[50];
    snprintf(temp, 50, "debug_out_%d", my_rank());
    debf = fopen(temp, "w");
    if (!debf) meep::abort("Unable to open debug output %s\n", temp);
  }
  vfprintf(debf, fmt, ap);
  fflush(debf);
  va_end(ap);
}

void master_fprintf(FILE *f, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  if (am_master()) {
    vfprintf(f, fmt, ap);
    fflush(f);
  }
  va_end(ap);
}
FILE *master_fopen(const char *name, const char *mode) {
  FILE *f = am_master() ? fopen(name, mode) : 0;

  /* other processes need to know if fopen returned zero, in order
     to abort if fopen failed.  If fopen was successfully, just return
     a random non-zero pointer (which is never used except to compare to zero)
     on non-master processes */
  if (broadcast(0, bool(f != 0)) && !am_master()) f = (FILE *)name;
  return f;
}
void master_fclose(FILE *f) {
  if (am_master()) fclose(f);
}

/* The following functions bracket a "critical section," a region
   of code that should be executed by only one process at a time.

   They work by having each process wait for a message from the
   previous process before starting.

   Each critical section is passed an integer "tag"...ideally, this
   should be a unique identifier for each critical section so that
   messages from different critical sections don't get mixed up
   somehow. */

void begin_critical_section(int tag) {
#ifdef HAVE_MPI
  int process_rank;
  MPI_Comm_rank(mycomm, &process_rank);
  if (process_rank > 0) { /* wait for a message before continuing */
    MPI_Status status;
    int recv_tag = tag - 1; /* initialize to wrong value */
    MPI_Recv(&recv_tag, 1, MPI_INT, process_rank - 1, tag, mycomm, &status);
    if (recv_tag != tag) meep::abort("invalid tag received in begin_critical_section");
  }
#else
  UNUSED(tag);
#endif
}

void end_critical_section(int tag) {
#ifdef HAVE_MPI
  int process_rank, num_procs;
  MPI_Comm_rank(mycomm, &process_rank);
  MPI_Comm_size(mycomm, &num_procs);
  if (process_rank != num_procs - 1) { /* send a message to next process */
    MPI_Send(&tag, 1, MPI_INT, process_rank + 1, tag, mycomm);
  }
#else
  UNUSED(tag);
#endif
}

/* Simple, somewhat hackish API to allow user to run multiple simulations
   in parallel in the same MPI job.  The user calls

   mygroup = divide_parallel_processes(numgroups);

   to divide all of the MPI processes into numgroups equal groups,
   and to return the index (from 0 to numgroups-1) of the current group.
   From this point on, all fields etc. that you create and all
   calls from mympi.cpp will only communicate within your group of
   processes.

   However, there are two calls that you can use to switch back to
   globally communication among all processes:

   begin_global_communications();
   ....do stuff....
   end_global_communications();

   It is important not to mix the two types; e.g. you cannot timestep
   a field created in the local group in global mode, or vice versa.
*/

int divide_parallel_processes(int numgroups) {
#ifdef HAVE_MPI
  end_divide_parallel();
  if (numgroups > count_processors()) meep::abort("numgroups > count_processors");
  int mygroup = (my_rank() * numgroups) / count_processors();
  MPI_Comm_split(MPI_COMM_WORLD, mygroup, my_rank(), &mycomm);
  return mygroup;
#else
  if (numgroups != 1) meep::abort("cannot divide processes in non-MPI mode");
  return 0;
#endif
}

#ifdef HAVE_MPI
static MPI_Comm mycomm_save = MPI_COMM_WORLD;
#endif

void begin_global_communications(void) {
#ifdef HAVE_MPI
  mycomm_save = mycomm;
  mycomm = MPI_COMM_WORLD;
#endif
}

void end_global_communications(void) {
#ifdef HAVE_MPI
  mycomm = mycomm_save;
  mycomm_save = MPI_COMM_WORLD;
#endif
}

void end_divide_parallel(void) {
#ifdef HAVE_MPI
  if (mycomm != MPI_COMM_WORLD) MPI_Comm_free(&mycomm);
  if (mycomm_save != MPI_COMM_WORLD) MPI_Comm_free(&mycomm_save);
  mycomm = mycomm_save = MPI_COMM_WORLD;
#endif
}

int my_global_rank() {
#ifdef HAVE_MPI
  if (!mpi_runtime_ready()) return 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
#else
  return 0;
#endif
}

} // namespace meep
