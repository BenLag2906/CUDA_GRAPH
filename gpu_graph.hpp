/* Benoit LAGADEC 

For more information refer to this code 

https://github.com/NVIDIA/cuda-samples

*/
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>

struct gpu_graph_t {

  enum class state_t { capture, update };

  void add_kernel_node(size_t key, cudaKernelNodeParams params, cudaStream_t s);

  void update_kernel_node(size_t key, cudaKernelNodeParams params);

  state_t state() { return _state; }

  ~gpu_graph_t();

private:

  std::unordered_map<size_t, cudaGraphNode_t> _node_map;

  state_t _state;

  cudaGraph_t _graph; /*!< The underlying cudaGraph_t object */

  cudaGraphExec_t _graph_exec; /*!< The underlying cudaGraphExec_t object */

  bool _instantiated = false; /*!< Whether _graph_exec has been instantiated */

  static void begin_capture(cudaStream_t s);

  void end_capture(cudaStream_t s);

  void launch_graph(cudaStream_t s);

public:

  bool _always_recapture = false;

  /**
    @brief If we have a valid graph ready, then (update and) launch it. If not, capture the graph.
   */
  template<class Obj>
  void wrap(Obj &o, cudaStream_t s) {
    if (!_always_recapture && _instantiated) {
      // If the graph has been instantiated, set the state to update
      _state = state_t::update;
      o(*this, s);
    } else {
      // If the graph has not been instantiated, set the state to capture
      _state = state_t::capture;
      begin_capture(s);
      o(*this, s);
      end_capture(s);
    }
    launch_graph(s);
  }

};
