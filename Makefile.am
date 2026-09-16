ACLOCAL_AMFLAGS=-I m4

SUBDIRS = src

if WITH_LIBCTL
if WITH_SCHEME
  SUBDIRS += scheme
endif
endif

if WITH_PYTHON
if WITH_MPB
if WITH_LIBCTLGEOM
  SUBDIRS += libpympb
endif
endif
  SUBDIRS += python
endif

# tests go last: they only depend on src/libmeep.la, so placing them
# after libpympb/python lets those directories start sooner with make -j
SUBDIRS += tests

# The Git snapshot does not contain maintainer-generated Scheme sources.
# Copy the optional interface source trees verbatim instead of invoking their
# dist hooks, which attempt to regenerate those files even when disabled.
DIST_SUBDIRS = src tests

EXTRA_DIST = NEWS.md LICENSE COPYRIGHT GPU_DEVELOPMENT.md m4 scheme python libpympb \
	cuda environment scripts meep-pkgconfig.in version.sh

dist-hook:
	test -n "$(distdir)" && test "$(distdir)" != "/" && \
		test -d "$(distdir)"
	rm -rf "$(distdir)/python/.deps" "$(distdir)/python/.libs" \
		"$(distdir)/python/meep" "$(distdir)/libpympb/.deps" \
		"$(distdir)/libpympb/.libs"
	rm -f "$(distdir)/python/Makefile" "$(distdir)/libpympb/Makefile" \
		"$(distdir)/python/"*.la "$(distdir)/python/"*.lo \
		"$(distdir)/python/"*.o "$(distdir)/libpympb/"*.la \
		"$(distdir)/libpympb/"*.lo "$(distdir)/libpympb/"*.o
	find "$(distdir)" -type f \( -name '*.pyc' -o -name '*.pyo' \) \
		-delete
	find "$(distdir)" -depth -type d -name __pycache__ -empty -delete

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = meep.pc

meep.pc: meep-pkgconfig
	cp -f $(top_builddir)/meep-pkgconfig $@

libmeep:
	cd src && $(MAKE)

# Generate the Python API Document
# PYTHONPATH is set to use the in-place build of meep
python_api_doc: all
	if test '$(top_srcdir)' != . ; then \
		cp -rf $(top_srcdir)/doc .; \
		chmod -R +rw doc; \
	fi
	if test -z $$SKIP_BUILD_DOCS; then \
		PYTHONPATH=./python $(PYTHON) doc/generate_py_api.py; \
	fi

# Build the HTML version of the docs
docs: python_api_doc
	if test -z $$SKIP_BUILD_DOCS; then \
		$(PYTHON) -m mkdocs build -f $(top_srcdir)/mkdocs.yml; \
	fi

# Build a tarball containing just the html docs
docs-dist: docs
	tar czvf meep-documentation-$(PACKAGE_VERSION).tgz doc/site

clean-local:
	rm -f meep-documentation-$(PACKAGE_VERSION).tgz
	if test '$(top_srcdir)' != . ; then \
		rm -rf doc; \
	fi

DISTCLEANFILES = meep.pc
