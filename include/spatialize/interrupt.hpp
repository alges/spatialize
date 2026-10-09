#ifndef _SPTLZ_INTERRUPT_
#define _SPTLZ_INTERRUPT_

#include <atomic>
#include <thread>
#include <pybind11/pybind11.h>

namespace sptlz{
  // Ctrl-C during a parallel loop. Python (signals) may only be touched by the thread that holds the
  // GIL, the calling thread, identified by its OS thread id. Any thread asks requested() between
  // units of work (cells): the calling thread then checks the signals, and every thread learns that
  // the loop must stop, so an interruption is answered within one cell, not one partition.
  struct Interrupt {
    const std::thread::id caller = std::this_thread::get_id();
    std::atomic<bool> stop{false};
    std::atomic<bool> signalled{false};

    bool requested(){
      if(!stop.load() && std::this_thread::get_id() == caller){
        if(PyErr_CheckSignals() != 0){  // to allow ctrl-c from user
          signalled.store(true);
          stop.store(true);
        }
      }
      return(stop.load());
    }
  };
}

#endif
