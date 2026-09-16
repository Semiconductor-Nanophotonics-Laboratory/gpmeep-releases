lib_LTLIBRARIES = libmeep.la
include_HEADERS = meep.hpp
pkginclude_HEADERS = meep/mympi.hpp meep/vec.hpp meep/gpu.hpp meep/meep-config.h meepgeom.hpp material_data.hpp adjust_verbosity.hpp

AM_CPPFLAGS = -I$(top_builddir)/src -I$(top_srcdir)/src -I$(top_srcdir)/cuda/include

BUILT_SOURCES = sphere-quad.h step_generic_stride1.cpp meep/meep-config.h

HDRS = meep.hpp meep_internals.hpp gpu_backend_internal.hpp gpu_grid_index.hpp meep/mympi.hpp meep/vec.hpp	\
bicgstab.hpp meep/gpu.hpp meepgeom.hpp material_data.hpp adjust_verbosity.hpp

libmeep_la_SOURCES = array_slice.cpp anisotropic_averaging.cpp 		\
bands.cpp boundaries.cpp bicgstab.cpp casimir.cpp 	\
cw_fields.cpp dft.cpp dft_ldos.cpp energy_and_flux.cpp 	\
fields.cpp fields_dump.cpp fix_boundary_sources.cpp gpu_backend.cpp loop_in_chunks.cpp h5fields.cpp h5file.cpp 	\
initialize.cpp integrate.cpp integrate2.cpp material_data.cpp monitor.cpp mympi.cpp 	\
multilevel-atom.cpp near2far.cpp output_directory.cpp random.cpp 	\
sources.cpp step.cpp step_db.cpp stress.cpp structure.cpp structure_dump.cpp		\
susceptibility.cpp time.cpp update_eh.cpp mpb.cpp update_pols.cpp 	\
vec.cpp step_generic.cpp meepgeom.cpp GDSIIgeom.cpp $(HDRS) $(BUILT_SOURCES)

if WITH_CUDA
libmeep_la_SOURCES += cuda_runtime.cu

CUDA_RUNTIME_FLAGS_STAMP = cuda-runtime-flags.stamp

# cuda_runtime.cu intentionally shares the standalone implementation through
# a textual include. Automake does not infer nvcc dependencies for the custom
# rule, so list every local input explicitly. The content-addressed stamp also
# makes architecture and math-flag changes invalidate the object. A repeated
# configure may conservatively rebuild once because it regenerates the
# architecture header; an unchanged repeated make does not.
cuda_runtime.lo: cuda_runtime.cu \
	$(top_srcdir)/cuda/src/runtime.cu \
	$(top_srcdir)/cuda/include/meep_cuda/runtime.hpp \
	$(top_srcdir)/cuda/include/meep_cuda/detail/bfast.hpp \
	$(top_srcdir)/cuda/include/meep_cuda/detail/cylindrical.hpp \
	$(top_srcdir)/cuda/include/meep_cuda/detail/curl.hpp \
	$(top_srcdir)/cuda/include/meep_cuda/detail/index_space.hpp \
	$(top_srcdir)/cuda/include/meep_cuda/detail/launch.hpp \
	$(top_srcdir)/cuda/include/meep_cuda/detail/polarization.hpp \
	$(top_srcdir)/cuda/include/meep_cuda/detail/update_eh.hpp \
	$(top_builddir)/src/meep_cuda/detail/compiled_architectures.hpp \
	$(CUDA_RUNTIME_FLAGS_STAMP)

$(CUDA_RUNTIME_FLAGS_STAMP): Makefile
	$(AM_V_GEN){ \
	  printf '%s\n' 'NVCC=$(NVCC)'; \
	  printf '%s\n' 'CUDAHOSTCXX=$(CUDAHOSTCXX)'; \
	  printf '%s\n' 'NVCCFLAGS=$(NVCCFLAGS)'; \
	  printf '%s\n' 'CUDA_ARCH_FLAGS=$(CUDA_ARCH_FLAGS)'; \
	} > $@.tmp
	$(AM_V_at)if test -r $@ && cmp -s $@.tmp $@; then \
	  rm -f $@.tmp; \
	else \
	  mv -f $@.tmp $@; \
	fi

CLEANFILES = $(CUDA_RUNTIME_FLAGS_STAMP) $(CUDA_RUNTIME_FLAGS_STAMP).tmp
endif

SUBDIRS = support
libmeep_la_LIBADD = support/libsupport.la $(CUDA_LIBS)

libmeep_la_LDFLAGS = -version-info @SHARED_VERSION_INFO@

SUFFIXES = .cu

.cu.lo:
	$(AM_V_CXX)$(LIBTOOL) --tag=CXX --mode=compile $(NVCC) \
	  $(DEFS) $(DEFAULT_INCLUDES) $(AM_CPPFLAGS) \
	  $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  --forward-unknown-to-host-compiler \
	  -std=c++14 -ccbin "$(CUDAHOSTCXX)" -Xcompiler=-fPIC \
	  -c $< -o $@

PRELUDE = "/* This file was automatically generated --- DO NOT EDIT */"

noinst_PROGRAMS = sphere_quad
sphere_quad_SOURCES = sphere-quad.cpp
SPHERE_QUAD = $(top_builddir)/src/sphere_quad
sphere-quad.h:
	make sphere_quad
	(echo $(PRELUDE); echo; $(SPHERE_QUAD)) > $@

step_generic_stride1.cpp: step_generic.cpp
	(echo $(PRELUDE); echo; \
	 sed -e 's/LOOP_OVER/S1LOOP_OVER/g' \
	     -e 's/step_curl/step_curl_stride1/g' \
	     -e 's/step_update_EDHB/step_update_EDHB_stride1/g' \
	     -e 's/step_beta/step_beta_stride1/g' \
	     -e 's/step_bfast/step_bfast_stride1/g' \
	     $(top_srcdir)/src/step_generic.cpp) > $@

MAINTAINERCLEANFILES = $(BUILT_SOURCES)
