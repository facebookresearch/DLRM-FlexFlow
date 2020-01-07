# $Source: bitbucket.org:berkeleylab/gasnet.git/other/fragment-body.mak.in $
# ----------------------------------------------------------------------
# Following section other/fragment-body.mak.  Generated from fragment-body.mak.in by configure.

# ----------------------------------------------------------------------
# Directory-based options

GASNET_INCLUDES =  -I###INSTALL_INCLUDE### -I###INSTALL_INCLUDE###/#conduit_name#-conduit $(CONDUIT_INCLUDES) $(CONDUIT_INCLUDES_#THREAD_MODEL#)
GASNET_LIBDIRS = -L###INSTALL_LIB###

# Textual lines containing the string "###NOINSTALL###" are removed by the install process
# (must be one continuous line) ###NOINSTALL###
GASNET_INCLUDES =  -I/private/home/ehsanardestani/DLRM_FlexFlow_h2/GASNet-2019.9.0 -I/private/home/ehsanardestani/DLRM_FlexFlow_h2/GASNet-2019.9.0/#conduit_name#-conduit -I/private/home/ehsanardestani/DLRM_FlexFlow_h2/GASNet-2019.9.0/other $(CONDUIT_INCLUDES) $(CONDUIT_INCLUDES_#THREAD_MODEL#) -I/private/home/ehsanardestani/DLRM_FlexFlow_h2/GASNet-2019.9.0/extended-ref/vis -I/private/home/ehsanardestani/DLRM_FlexFlow_h2/GASNet-2019.9.0/extended-ref/coll -I/private/home/ehsanardestani/DLRM_FlexFlow_h2/GASNet-2019.9.0/extended-ref/ratomic -I/private/home/ehsanardestani/DLRM_FlexFlow_h2/GASNet-2019.9.0/extended-ref -I/private/home/ehsanardestani/DLRM_FlexFlow_h2/GASNet-2019.9.0  ###NOINSTALL###
GASNET_LIBDIRS = -L/private/home/ehsanardestani/DLRM_FlexFlow_h2/GASNet-2019.9.0/#conduit_name#-conduit ###NOINSTALL###

# ----------------------------------------------------------------------
# C compiler and options

GASNET_CC = /usr/bin/gcc

GASNET_OPT_CFLAGS = -O3 --param max-inline-insns-single=35000 --param inline-unit-growth=10000 --param large-function-growth=200000 $(CONDUIT_OPT_CFLAGS) $(CONDUIT_OPT_CFLAGS_#THREAD_MODEL#)
GASNET_MISC_CFLAGS =  -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address $(CONDUIT_MISC_CFLAGS) $(CONDUIT_MISC_CFLAGS_#THREAD_MODEL#)
GASNET_MISC_CPPFLAGS =  $(CONDUIT_MISC_CPPFLAGS) $(CONDUIT_MISC_CPPFLAGS_#THREAD_MODEL#)
GASNET_DEVWARN_CFLAGS =  -Winline -Wall -Wpointer-arith -Wnested-externs -Wwrite-strings -Wmissing-format-attribute -Winit-self -Wvla -Wexpansion-to-defined -Woverlength-strings -Wclobbered -Wempty-body -Wignored-qualifiers -Wimplicit-fallthrough -Wmissing-parameter-type -Wold-style-declaration -Wuninitialized -Wshift-negative-value -Wno-format-overflow -Wno-format-truncation $(CONDUIT_DEVWARN_CFLAGS) $(CONDUIT_DEVWARN_CFLAGS#THREAD_MODEL#)

# ----------------------------------------------------------------------
# C++ compiler and options
# TODO: some options (especially CONDUIT_*) are not distinct from C compiler

GASNET_CXX = /usr/bin/g++

GASNET_OPT_CXXFLAGS = -O2 $(CONDUIT_OPT_CFLAGS) $(CONDUIT_OPT_CFLAGS_#THREAD_MODEL#)
GASNET_MISC_CXXFLAGS =  -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address $(CONDUIT_MISC_CFLAGS) $(CONDUIT_MISC_CFLAGS_#THREAD_MODEL#)
GASNET_MISC_CXXCPPFLAGS =  $(CONDUIT_MISC_CPPFLAGS) $(CONDUIT_MISC_CPPFLAGS_#THREAD_MODEL#)
GASNET_DEVWARN_CXXFLAGS =  -Wall -Wpointer-arith -Wwrite-strings -Wmissing-format-attribute -Winit-self -Wvla -Wexpansion-to-defined -Woverlength-strings -Wclobbered -Wempty-body -Wignored-qualifiers -Wimplicit-fallthrough -Wuninitialized -Wshift-negative-value -Wno-format-overflow -Wno-format-truncation $(CONDUIT_DEVWARN_CXXFLAGS) $(CONDUIT_DEVWARN_CXXFLAGS#THREAD_MODEL#)

# ----------------------------------------------------------------------
# Common defines

GASNET_EXTRADEFINES_SEQ = 
GASNET_EXTRADEFINES_PAR = -D_REENTRANT
GASNET_EXTRADEFINES_PARSYNC = -D_REENTRANT

GASNET_DEFINES = -D_GNU_SOURCE=1 -DGASNET_#THREAD_MODEL# $(GASNET_EXTRADEFINES_#THREAD_MODEL#) $(CONDUIT_DEFINES) $(CONDUIT_DEFINES_#THREAD_MODEL#) $(MANUAL_DEFINES)

# ----------------------------------------------------------------------
# Documented compilation convenience aliases

GASNET_CFLAGS = $(GASNET_OPT_CFLAGS) $(GASNET_MISC_CFLAGS) $(MANUAL_CFLAGS)
GASNET_CPPFLAGS = $(GASNET_MISC_CPPFLAGS) $(GASNET_DEFINES) $(GASNET_INCLUDES)

GASNET_CXXFLAGS = $(GASNET_OPT_CXXFLAGS) $(GASNET_MISC_CXXFLAGS) $(MANUAL_CXXFLAGS)
GASNET_CXXCPPFLAGS = $(GASNET_MISC_CXXCPPFLAGS) $(GASNET_DEFINES) $(GASNET_INCLUDES)

# ----------------------------------------------------------------------
# linker and options

GASNET_LD = $(GASNET_LD_OVERRIDE)

# linker flags that GASNet clients should use 
GASNET_LDFLAGS = $(GASNET_LDFLAGS_OVERRIDE)  $(CONDUIT_LDFLAGS) $(CONDUIT_LDFLAGS_#THREAD_MODEL#) $(MANUAL_LDFLAGS)

GASNET_EXTRALIBS_SEQ = 
GASNET_EXTRALIBS_PAR = -lpthread
GASNET_EXTRALIBS_PARSYNC = -lpthread

# libraries that GASNet clients should append to link line
GASNET_LIBS =                             \
    $(GASNET_LIBDIRS)                     \
    -lgasnet-#conduit_name#-#thread_model# \
    $(CONDUIT_LIBS)                       \
    $(CONDUIT_LIBS_#THREAD_MODEL#)        \
    $(GASNET_EXTRALIBS_#THREAD_MODEL#)    \
                        \
    -L/usr/lib/gcc/x86_64-linux-gnu/7 -lgcc                              \
                                    \
    -lm                                \
    $(MANUAL_LIBS)

# ----------------------------------------------------------------------
