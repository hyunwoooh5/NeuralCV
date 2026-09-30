CXX ?= c++
CXXFLAGS ?= -O3 -std=c++11 -DNDEBUG
EIGEN_INCLUDE ?= $(HOME)/eigen-3.3.9
CLI11_INCLUDE ?= $(HOME)/local/include
CPPFLAGS += -I$(EIGEN_INCLUDE) -I$(CLI11_INCLUDE) -I$(HOME)/Programs/include -I$(HOME)/local/include -I.
LDLIBS += -lm

.PHONY: clean

sample_%: sample_%.cpp monte_carlo.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f sample_scalar_2d sample_scalar_3d sample_scalar_4d \
		sample_su2_2d_obc_bronzan sample_su2_2d_obc_euler \
		sample_transmon sample_u1_2d_obc sample_u1_2d_pbc
