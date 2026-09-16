#define _GNU_SOURCE 1

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#define GPMEEP_FD_ALLOCATION_ABI 1U
#define GPMEEP_EXPORT __attribute__((visibility("default")))

struct gpmeep_fd_result {
  uint32_t abi_version;
  int32_t completed;
  int32_t fd;
  int32_t error_number;
};

struct gpmeep_pipe_result {
  uint32_t abi_version;
  int32_t completed;
  int32_t read_fd;
  int32_t write_fd;
  int32_t error_number;
};

_Static_assert(sizeof(struct gpmeep_fd_result) == 16,
               "gpmeep_fd_result ABI size changed");
_Static_assert(offsetof(struct gpmeep_fd_result, abi_version) == 0,
               "gpmeep_fd_result abi_version offset changed");
_Static_assert(offsetof(struct gpmeep_fd_result, completed) == 4,
               "gpmeep_fd_result completed offset changed");
_Static_assert(offsetof(struct gpmeep_fd_result, fd) == 8,
               "gpmeep_fd_result fd offset changed");
_Static_assert(offsetof(struct gpmeep_fd_result, error_number) == 12,
               "gpmeep_fd_result error_number offset changed");
_Static_assert(sizeof(struct gpmeep_pipe_result) == 20,
               "gpmeep_pipe_result ABI size changed");
_Static_assert(offsetof(struct gpmeep_pipe_result, abi_version) == 0,
               "gpmeep_pipe_result abi_version offset changed");
_Static_assert(offsetof(struct gpmeep_pipe_result, completed) == 4,
               "gpmeep_pipe_result completed offset changed");
_Static_assert(offsetof(struct gpmeep_pipe_result, read_fd) == 8,
               "gpmeep_pipe_result read_fd offset changed");
_Static_assert(offsetof(struct gpmeep_pipe_result, write_fd) == 12,
               "gpmeep_pipe_result write_fd offset changed");
_Static_assert(offsetof(struct gpmeep_pipe_result, error_number) == 16,
               "gpmeep_pipe_result error_number offset changed");

static void gpmeep_fd_result_begin(struct gpmeep_fd_result *result) {
  result->abi_version = GPMEEP_FD_ALLOCATION_ABI;
  __atomic_store_n(&result->completed, 0, __ATOMIC_RELAXED);
  result->fd = -1;
  result->error_number = 0;
}

static void gpmeep_fd_result_finish(struct gpmeep_fd_result *result, int fd,
                                    int error_number) {
  result->fd = fd;
  result->error_number = error_number;
  /* Publish completion last so Python can recover ownership even when an
     exception is dispatched immediately after the native call returns. */
  __atomic_store_n(&result->completed, 1, __ATOMIC_RELEASE);
}

static void gpmeep_pipe_result_begin(struct gpmeep_pipe_result *result) {
  result->abi_version = GPMEEP_FD_ALLOCATION_ABI;
  __atomic_store_n(&result->completed, 0, __ATOMIC_RELAXED);
  result->read_fd = -1;
  result->write_fd = -1;
  result->error_number = 0;
}

static void gpmeep_pipe_result_finish(struct gpmeep_pipe_result *result,
                                      int read_fd, int write_fd,
                                      int error_number) {
  result->read_fd = read_fd;
  result->write_fd = write_fd;
  result->error_number = error_number;
  __atomic_store_n(&result->completed, 1, __ATOMIC_RELEASE);
}

GPMEEP_EXPORT uint32_t gpmeep_fd_allocation_shim_abi(void) {
  return GPMEEP_FD_ALLOCATION_ABI;
}

GPMEEP_EXPORT uint64_t gpmeep_fd_result_layout(void) {
  return ((uint64_t)sizeof(struct gpmeep_fd_result) << 32) |
         ((uint64_t)offsetof(struct gpmeep_fd_result, abi_version) << 24) |
         ((uint64_t)offsetof(struct gpmeep_fd_result, completed) << 16) |
         ((uint64_t)offsetof(struct gpmeep_fd_result, fd) << 8) |
         (uint64_t)offsetof(struct gpmeep_fd_result, error_number);
}

GPMEEP_EXPORT uint64_t gpmeep_pipe_result_layout(void) {
  return ((uint64_t)sizeof(struct gpmeep_pipe_result) << 40) |
         ((uint64_t)offsetof(struct gpmeep_pipe_result, abi_version) << 32) |
         ((uint64_t)offsetof(struct gpmeep_pipe_result, completed) << 24) |
         ((uint64_t)offsetof(struct gpmeep_pipe_result, read_fd) << 16) |
         ((uint64_t)offsetof(struct gpmeep_pipe_result, write_fd) << 8) |
         (uint64_t)offsetof(struct gpmeep_pipe_result, error_number);
}

GPMEEP_EXPORT void
gpmeep_dupfd_cloexec_into(int source_fd, int minimum_fd,
                          struct gpmeep_fd_result *result) {
  int fd;
  int saved_errno;

  if (result == NULL)
    return;
  gpmeep_fd_result_begin(result);
  errno = 0;
  fd = fcntl(source_fd, F_DUPFD_CLOEXEC, minimum_fd);
  saved_errno = fd < 0 ? errno : 0;
  gpmeep_fd_result_finish(result, fd, saved_errno);
}

GPMEEP_EXPORT void gpmeep_pidfd_open_into(pid_t pid, unsigned int flags,
                                           struct gpmeep_fd_result *result) {
  int fd;
  int saved_errno;

  if (result == NULL)
    return;
  gpmeep_fd_result_begin(result);
  errno = 0;
#ifdef SYS_pidfd_open
  fd = (int)syscall(SYS_pidfd_open, pid, flags);
#else
  errno = ENOSYS;
  fd = -1;
#endif
  saved_errno = fd < 0 ? errno : 0;
  gpmeep_fd_result_finish(result, fd, saved_errno);
}

GPMEEP_EXPORT void gpmeep_pipe2_into(int flags,
                                     struct gpmeep_pipe_result *result) {
  int descriptors[2] = {-1, -1};
  int status;
  int saved_errno;

  if (result == NULL)
    return;
  gpmeep_pipe_result_begin(result);
  errno = 0;
  status = pipe2(descriptors, flags);
  saved_errno = status < 0 ? errno : 0;
  if (status < 0) {
    descriptors[0] = -1;
    descriptors[1] = -1;
  }
  gpmeep_pipe_result_finish(result, descriptors[0], descriptors[1],
                            saved_errno);
}
