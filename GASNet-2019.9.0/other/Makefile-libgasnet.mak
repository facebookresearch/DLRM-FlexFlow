# This Makefile fragment is used to build GASNet libraries
# it is not meant to be used directly 
# other/Makefile-libgasnet.mak.  Generated from Makefile-libgasnet.mak.in by configure.

.PHONY: do-libgasnet-seq do-libgasnet-par do-libgasnet-parsync  \
        do-libgasnet_tools-seq do-libgasnet_tools-par do-libgasnet_tools  \
        do-libgasnet check-exports do-pthreads-error do-tools-make-fragment



thread_defines = -D_REENTRANT
SEPARATE_CC = 

PLPA_INCLUDES = -I$(top_srcdir)/other/plpa/src/libplpa
PLPA_SOURCES  = $(top_srcdir)/other/plpa/src/libplpa/plpa_api_probe.c \
                $(top_srcdir)/other/plpa/src/libplpa/plpa_dispatch.c

#PSHM_SOURCES  = $(top_srcdir)/gasnet_pshm.c

#PTHREADS_ERROR_CHECK = $(MAKE) do-pthreads-error
PTHREADS_ERROR_CHECK = :

TOOLLIBINCLUDES =         \
          \
        -I$(srcdir)       \
        -I$(top_srcdir)   \
        -I$(top_builddir) \
        -I$(top_srcdir)/other \
	$(PLPA_INCLUDES)

LIBINCLUDES = $(TOOLLIBINCLUDES) \
        -I$(top_srcdir)/extended-ref/coll \
        -I$(top_srcdir)/extended-ref/vis \
        -I$(top_srcdir)/extended-ref/ratomic \
        -I$(top_srcdir)/extended-ref  


TOOLLIBDEFINES =                    \
        -D_GNU_SOURCE=1      \
        $(LIBGASNET_THREAD_DEFINES) \
        $(MANUAL_DEFINES)

LIBDEFINES =                        \
	$(TOOLLIBDEFINES)	    \
        -DGASNET_$(THREAD_MODEL)    

 TOOLLIB_DEBUGFLAGS = -DNDEBUG
TOOLLIBCFLAGS =                   \
        -DGASNETT_BUILDING_TOOLS  \
        -O3 --param max-inline-insns-single=35000 --param inline-unit-growth=10000 --param large-function-growth=200000       \
	 -Winline -Wall -Wpointer-arith -Wnested-externs -Wwrite-strings -Wmissing-format-attribute -Winit-self -Wvla -Wexpansion-to-defined -Woverlength-strings -Wclobbered -Wempty-body -Wignored-qualifiers -Wimplicit-fallthrough -Wmissing-parameter-type -Wold-style-declaration -Wuninitialized -Wshift-negative-value -Wno-format-overflow -Wno-format-truncation          \
         -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address             \
                   \
        $(TOOLLIBDEFINES)         \
	$(TOOLLIB_DEBUGFLAGS)     \
	$(TOOLLIBINCLUDES)	  \
	$${keeptmps:+-save-temps} \
	$(MANUAL_CFLAGS)

LIBCFLAGS =                       \
        -O3 --param max-inline-insns-single=35000 --param inline-unit-growth=10000 --param large-function-growth=200000       \
	 -Winline -Wall -Wpointer-arith -Wnested-externs -Wwrite-strings -Wmissing-format-attribute -Winit-self -Wvla -Wexpansion-to-defined -Woverlength-strings -Wclobbered -Wempty-body -Wignored-qualifiers -Wimplicit-fallthrough -Wmissing-parameter-type -Wold-style-declaration -Wuninitialized -Wshift-negative-value -Wno-format-overflow -Wno-format-truncation          \
         -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address             \
                   \
              \
        $(LIBDEFINES)             \
        $(CONDUIT_EXTRALIBCFLAGS) \
	$(LIBINCLUDES)		  \
	$${keeptmps:+-save-temps} \
	$(MANUAL_CFLAGS)

libgasnet_tools_sources =            \
	$(top_srcdir)/gasnet_tools.c \
	$(PLPA_SOURCES)

libgasnet_sources =                                         \
        $(CONDUIT_SOURCELIST)                               \
        $(libgasnet_tools_sources)                          \
        $(PSHM_SOURCES)                                     \
        $(top_srcdir)/extended-ref/vis/gasnet_refvis.c      \
        $(top_srcdir)/extended-ref/ratomic/gasnet_refratomic.c\
        $(top_srcdir)/extended-ref/coll/gasnet_refcoll.c    \
        $(top_srcdir)/extended-ref/coll/gasnet_putget.c     \
        $(top_srcdir)/extended-ref/coll/gasnet_eager.c      \
        $(top_srcdir)/extended-ref/coll/gasnet_rvous.c      \
        $(top_srcdir)/extended-ref/coll/gasnet_team.c	    \
        $(top_srcdir)/extended-ref/coll/gasnet_hashtable.c  \
        $(top_srcdir)/extended-ref/coll/gasnet_reduce.c     \
        $(top_srcdir)/gasnet_event.c                        \
        $(top_srcdir)/gasnet_legacy.c                       \
        $(top_srcdir)/gasnet_internal.c                     \
        $(top_srcdir)/gasnet_am.c                           \
        $(top_srcdir)/gasnet_trace.c                        \
        $(top_srcdir)/gasnet_mmap.c                         \
        $(top_srcdir)/gasnet_tm.c                           \
	$(top_srcdir)/gasnet_diagnostic.c

libgasnet_objects = \
	`for file in $(libgasnet_sources) ; do echo \`basename $$file .c\`.o ; done` \
	$(CONDUIT_SPECIAL_OBJS)

libgasnet_tools_dependencies =  \
        $(CONFIG_HEADER)        \
        $(top_srcdir)/*.[ch]    \
        $(top_srcdir)/other/*.h

libgasnet_dependencies =                  \
        $(libgasnet_tools_dependencies)   \
        $(srcdir)/*.[ch]                  \
        $(top_srcdir)/extended-ref/*/*.[ch] \
        $(top_srcdir)/extended-ref/*.[ch] \
        $(top_srcdir)/tests/test.h        \
	$(CONDUIT_SOURCELIST)             \
	$(CONDUIT_EXTRAHEADERS)           \
	$(CONDUIT_EXTRADEPS)

# library targets 
THREAD_MODEL=SEQ
THREAD_MODEL_LC=`echo "$(THREAD_MODEL)" | gawk '{print tolower($$0)}'`
LIBGASNET_NAME=libgasnet-$(CONDUIT_NAME)
do-libgasnet: $(CONDUIT_SPECIAL_OBJS)
	@mkdir -p .$(THREAD_MODEL)
	@libgasnet_objects="$(libgasnet_objects)" ; libgasnet_objects=`echo $$libgasnet_objects` ; \
	pwd=`/bin/pwd`; keeptmps='$(KEEPTMPS)'; \
	if test -z '$(KEEPTMPS)'; then rmcmd="&& rm -f $$libgasnet_objects"; fi; \
	if test -n '$(SEPARATE_CC)' ; then \
	  compcmd="for file in $(libgasnet_sources) ; do $(CC) $(LIBCFLAGS) -c "'$$file'" || exit "'$$?'" ; done" ; \
	else \
	  compcmd="$(CC) $(LIBCFLAGS) -c $(libgasnet_sources)" ; \
	fi ; \
	cmd="$$compcmd && \
	$(AR) cru $$pwd/$(LIBGASNET_NAME)-$(THREAD_MODEL_LC).a $$libgasnet_objects && \
	$(RANLIB) $$pwd/$(LIBGASNET_NAME)-$(THREAD_MODEL_LC).a $$rmcmd"; \
	echo " --- BUILDING $(LIBGASNET_NAME)-$(THREAD_MODEL_LC).a --- " ; \
        echo $$cmd ; cd .$(THREAD_MODEL) ; eval $$cmd
	@test -n '$(KEEPTMPS)' || rm -Rf .$(THREAD_MODEL)

set_dirs = top_srcdir=`cd $(top_srcdir); /bin/pwd`         \
           top_builddir=`cd $(top_builddir); /bin/pwd`     \
           srcdir=`cd $(srcdir); /bin/pwd`                 \
           builddir=`/bin/pwd`                             

do-libgasnet-seq: $(libgasnet_dependencies) $(CONDUIT_SEQ_HOOK)
	@$(MAKE) THREAD_MODEL=SEQ                            \
	  LIBGASNET_THREAD_DEFINES=                          \
          $(set_dirs) do-libgasnet

do-libgasnet-par: $(libgasnet_dependencies) $(CONDUIT_PAR_HOOK)
	@$(PTHREADS_ERROR_CHECK)
	@$(MAKE) THREAD_MODEL=PAR                            \
          LIBGASNET_THREAD_DEFINES="$(thread_defines)"       \
          $(set_dirs) do-libgasnet

do-libgasnet-parsync: $(libgasnet_dependencies) $(CONDUIT_PARSYNC_HOOK)
	@$(PTHREADS_ERROR_CHECK)
	@$(MAKE) THREAD_MODEL=PARSYNC                        \
          LIBGASNET_THREAD_DEFINES="$(thread_defines)"       \
          $(set_dirs) do-libgasnet

do-libgasnet_tools:
	@keeptmps="$(KEEPTMPS)" ;                            \
	 $(MAKE)                                             \
	  LIBCFLAGS="$(TOOLLIBCFLAGS)"                       \
          LIBGASNET_NAME=libgasnet_tools		     \
	  libgasnet_sources="$(libgasnet_tools_sources)"     \
          do-libgasnet

do-libgasnet_tools-seq: $(libgasnet_tools_dependencies)
	@$(MAKE) THREAD_MODEL=SEQ                            \
	  LIBGASNET_THREAD_DEFINES=-DGASNETT_THREAD_SINGLE   \
          $(set_dirs) do-libgasnet_tools

do-libgasnet_tools-par: $(libgasnet_tools_dependencies)
	@$(PTHREADS_ERROR_CHECK)
	@$(MAKE) THREAD_MODEL=PAR                            \
          LIBGASNET_THREAD_DEFINES="$(thread_defines) -DGASNETT_THREAD_SAFE" \
          $(set_dirs) do-libgasnet_tools

fragment_deps =  $(top_builddir)/other/gasnet_tools-fragment.mak

$(top_builddir)/other/gasnet_tools-fragment.mak: $(top_srcdir)/other/gasnet_tools-fragment.mak.in
	cd "$(top_builddir)/other" && $(MAKE) gasnet_tools-fragment.mak	

gasnet_tools-par.mak : $(fragment_deps)
	@$(PTHREADS_ERROR_CHECK)
	$(MAKE) do-tools-make-fragment thread_model=par THREAD_MODEL=PAR

gasnet_tools-seq.mak: $(fragment_deps)
	$(MAKE) do-tools-make-fragment thread_model=seq THREAD_MODEL=SEQ

do-tools-make-fragment: force
	rm -f gasnet_tools-$(thread_model).mak
	@echo Building gasnet_tools-$(thread_model).mak... ;                             \
        AUTOGENMSG='WARNING: This file is automatically generated - do NOT edit directly' ; \
        cat $(top_builddir)/other/gasnet_tools-fragment.mak |                               \
        sed -e 's@#THREAD_MODEL#@$(THREAD_MODEL)@g'                                         \
            -e 's@#thread_model#@$(thread_model)@g'                                         \
            -e "s@#AUTOGEN#@$${AUTOGENMSG}@g"                                               \
        > gasnet_tools-$(thread_model).mak || exit 1

pkgconfig_tools = $(top_srcdir)/other/pkgconfig-tools.pc

gasnet_tools-seq.pc: gasnet_tools-seq.mak $(pkgconfig_tools)
	@$(MAKE) do-pkgconfig-tools thread_model=seq pkgconfig_file="$@" FRAGMENT="$<"

gasnet_tools-par.pc: gasnet_tools-par.mak $(pkgconfig_tools)
	@$(PTHREADS_ERROR_CHECK)
	@$(MAKE) do-pkgconfig-tools thread_model=par pkgconfig_file="$@" FRAGMENT="$<"

do-pkgconfig-tools: force
	rm -f $(pkgconfig_file)
	@echo Building $(pkgconfig_file) from $$FRAGMENT...
	@echo '# WARNING: This file is automatically generated - do NOT edit directly' > $(pkgconfig_file)
	@echo '# Copyright 2017, The Regents of the University of California' >> $(pkgconfig_file)
	@echo '# Terms of use are as specified in license.txt' >> $(pkgconfig_file)
	@echo '# See the GASNet README for instructions on using these variables' >> $(pkgconfig_file)
	@VARS="GASNETTOOLS_CC GASNETTOOLS_CPPFLAGS GASNETTOOLS_CFLAGS \
               GASNETTOOLS_CXX GASNETTOOLS_CXXFLAGS \
               GASNETTOOLS_LD GASNETTOOLS_LDFLAGS GASNETTOOLS_LIBS" ; \
           $(MAKE) --no-print-directory -f $(top_srcdir)/other/Makefile-echovar.mak VARS="$$VARS" echovars \
	           >> $(pkgconfig_file)
	@cat $(pkgconfig_tools) | \
        sed -e 's@#thread_model#@$(thread_model)@g'  \
            -e 's@#version#@$(VERSION)@g'            \
        >> $(pkgconfig_file)

do-pthreads-error: 
	@echo "ERROR: pthreads support was not detected at configure time"
	@echo "       try re-running configure with --enable-pthreads"
	@exit 1

# bug1613: avoid automake infinite recursion here, because top-level Makefile includes this
# fragment and also provides the rules for rebuilding config.status
#cd $(top_builddir)/other && $(MAKE) Makefile-libgasnet.mak
$(top_builddir)/other/Makefile-libgasnet.mak: $(top_srcdir)/other/Makefile-libgasnet.mak.in
	cd $(top_builddir) && CONFIG_FILES=other/Makefile-libgasnet.mak CONFIG_HEADERS= ./config.status

check-exports: $(libraries)
	@echo Checking libgasnet exports...
	@if test x$(CHECK_EXPORTS) = x0; then                                       \
	  echo Skipped by user request ;                                            \
	  exit 0 ;                                                                  \
	 fi ;                                                                       \
	 failed=0 ;                                                                 \
	 for lib in $(libraries) ; do                                               \
	  echo ;                                                                    \
	  echo $$lib: ;                                                             \
	  /usr/bin/nm --defined-only $$lib |                                               \
	    /usr/bin/perl -pe 's/ \.refptr\.//' |                                          \
	    grep -v -e ' [\._]*gasnet_' -e ' [\._]*gasnet[tiecX]_' -e ' [\._]*gex_' \
		    -e ' [\._]*fh_' -e ' [\._]*firehose_'                           \
		    -e ' [\._]*fh[ic]_' -e ' [\._]*fhsmp_' -e ' [\._]*fhuni_'       \
		    -e ' [\._]*myxml_' -e ' [\._]*smp_coll_'                        \
		    -e ' [\._]*emutls_' -e 'get_pc_thunk'                           \
		    -e ' D bg_[a-z]' -e ' D _uci_' -e ' D _parse[A-Z][a-z]'         \
		    -e __FUNCTION__ -e __PRETTY_FUNCTION__ -e ' [\._]*DWinfo'       \
               -e ' [\._][\._]*debug_'                                              \
               -e ' [\._]*stab' -e ' [\._]*gnu_dev_' -e '^00* W '  |           \
	    /usr/bin/perl -n -e 'print if /^[0-9a-fA-F]+\s+[A-Z]\s+/' > .$$lib.exp;        \
	  if test -s .$$lib.exp ; then                                              \
	    cat .$$lib.exp ;                                                        \
	    echo FAILED ;                                                           \
	    failed=1 ;                                                              \
	  else                                                                      \
	    echo PASSED ;                                                           \
	  fi ;                                                                      \
	  rm -f .$$lib.exp ;                                                        \
	done ; exit $$failed

#check-exports: $(libraries)
#	@echo check-exports test SKIPPED

check-pkgconfig:
	@echo Checking pkgconfig...
	@if test x$(CHECK_PKGCONFIG) = x0; then                            \
	  echo SKIPPED: by user request ;                                  \
	  exit 0 ;                                                         \
	 elif ! pkg-config --atleast-pkgconfig-version=0.16.0 > /dev/null ; then \
	  echo SKIPPED: No working pkg-config found ;                      \
	  exit 0 ;                                                         \
	 fi ;                                                              \
	 validate=--validate ;                                             \
	 if test -z "`pkg-config --help 2>&1 | grep -e --validate`" ; then \
	   validate=--cflags ;                                             \
	 fi ;                                                              \
	 for file in $(CHECK_FILES) ; do                                   \
	   echo ;                                                          \
	   echo $$file: ;                                                  \
	   if pkg-config $$validate $$file ; then                          \
	     echo PASSED ;                                                 \
	   else                                                            \
	     echo FAILED ; exit 1 ;                                        \
	   fi ;                                                            \
	 done
