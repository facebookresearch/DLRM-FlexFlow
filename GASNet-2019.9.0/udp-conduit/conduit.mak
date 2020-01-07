#INSTRUCTIONS# Conduit-specific Makefile fragment settings
#INSTRUCTIONS#
#INSTRUCTIONS# The contents of this file are embedded into the 
#INSTRUCTIONS# *-(seq,par,parsync).mak Makefile fragments at conduit build time
#INSTRUCTIONS# The settings in those fragments are used to build GASNet clients
#INSTRUCTIONS# (including the GASNet tests). 
#INSTRUCTIONS# See the conduit-writer instructions in the generated fragments
#INSTRUCTIONS# or $(top_srcdir)/other/fragment-head.mak.in for usage info.

# AMUDP is C++-based, which requires us to link using the C++ compiler
GASNET_LD_OVERRIDE = /usr/bin/g++
GASNET_LDFLAGS_OVERRIDE = -O2  -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address  

# Linker feature requirements embedded in GASNET_LD(FLAGS) which are not satisfied solely by GASNET_LIBS 
# (eg possible dependence on implicit MPI or C++ libraries added by a linker wrapper in GASNET_LD):
GASNET_LD_REQUIRES_CXX = 1

# hooks for using AMUDP library from within build tree ###NOINSTALL###
# (nothing additional required for installed copy)     ###NOINSTALL###
CONDUIT_INCLUDES = -I/private/home/ehsanardestani/DLRM_FlexFlow_h2/GASNet-2019.9.0/other/amudp -I/private/home/ehsanardestani/DLRM_FlexFlow_h2/GASNet-2019.9.0/other/amudp ###NOINSTALL###

CONDUIT_LDFLAGS = 
CONDUIT_LIBDIRS = -L/private/home/ehsanardestani/DLRM_FlexFlow_h2/GASNet-2019.9.0/other/amudp ###NOINSTALL###

CONDUIT_LIBS = $(CONDUIT_LIBDIRS) -lamudp    

