#ifndef _SPTLZ_INTERRUPT_
#define _SPTLZ_INTERRUPT_

#include <atomic>
#include <thread>
#include <functional>
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
    // run by the calling thread at each check, e.g. to report the progress of every thread
    std::function<void()> on_check;

    bool requested(){
      if(!stop.load() && std::this_thread::get_id() == caller){
        if(PyErr_CheckSignals() != 0){  // to allow ctrl-c from user
          signalled.store(true);
          stop.store(true);
        }else if(on_check){
          on_check();
        }
      }
      return(stop.load());
    }
  };
}

#endif
