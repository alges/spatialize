#ifndef _SPTLZ_EMPTY_CELLS_
#define _SPTLZ_EMPTY_CELLS_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sptlz{
  // What a member is when its cell holds no datum (the session setting empty_cells):
  //   NAN      the member is NaN; the reading is the law conditioned on the cell having data;
  //   MARK     the cell receives one value, shared by every location in it (the theory's limit law
  //            sends the residual weight 1 - sum w_i to the mark law, and one shared mark per cell
  //            keeps the joint law of the locations right);
  //   COARSEN  the decoder runs on the nearest coarser cell that holds data.
  // Under MARK the value comes from a cell holding data, drawn among (mark_source)
  //   LOCAL    the mark_knn cells with data nearest to the empty cell (by their leaf points),
  //   CELLS    every cell with data of the partition (one vote per cell, the block-mark model),
  //   DATA     no cell: one datum drawn among all the data;
  // and the drawn cell gives (mark_value) either its decoder's prediction at the empty cell's point
  // (DECODER: the kind of value the decoder gives elsewhere, an observed value for the drawing
  // decoders) or one of its data drawn uniformly (DATUM: the block-mark model, where a block's value
  // is one mark whatever the decoder). The
  // draws are independent, with repetition, so the marks of distinct empty cells are iid given the
  // partition. In leave-one-out and k-fold the held-out data take no part.
  enum class EmptyCells { NAN_MEMBER, MARK, COARSEN };
  enum class MarkSource { LOCAL, CELLS, DATA };

  struct EmptyCellPolicy {
    EmptyCells kind = EmptyCells::NAN_MEMBER;
    MarkSource source = MarkSource::LOCAL;
    int knn = 8;
    bool value_from_decoder = true;

    static EmptyCellPolicy from(const std::string &policy, const std::string &mark_source, int mark_knn,
                                const std::string &mark_value){
      EmptyCellPolicy p;
      if(policy == "nan") p.kind = EmptyCells::NAN_MEMBER;
      else if(policy == "mark") p.kind = EmptyCells::MARK;
      else if(policy == "coarsen") p.kind = EmptyCells::COARSEN;
      else throw std::runtime_error("empty_cells must be 'nan', 'mark' or 'coarsen'; got '" + policy + "'");
      if(mark_source == "local") p.source = MarkSource::LOCAL;
      else if(mark_source == "cells") p.source = MarkSource::CELLS;
      else if(mark_source == "data") p.source = MarkSource::DATA;
      else throw std::runtime_error("mark_source must be 'local', 'cells' or 'data'; got '" + mark_source + "'");
      if(mark_knn < 1) throw std::runtime_error("mark_knn must be a positive integer");
      p.knn = mark_knn;
      if(mark_value == "decoder") p.value_from_decoder = true;
      else if(mark_value == "datum") p.value_from_decoder = false;
      else throw std::runtime_error("mark_value must be 'decoder' or 'datum'; got '" + mark_value + "'");
      return(p);
    }
  };

  // counter-based uniform in [0, 1) from (seed, tree, key, stage) alone, so a mark depends neither
  // on the number of threads nor on the other locations
  inline uint64_t mark_mix(uint64_t x){
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return(x ^ (x >> 31));
  }

  inline double mark_uniform(unsigned int seed, int tree, uint64_t key, uint64_t stage){
    uint64_t h = mark_mix(0xE3C7A9ULL ^ static_cast<uint64_t>(seed));
    h = mark_mix(h ^ static_cast<uint64_t>(static_cast<uint32_t>(tree)));
    h = mark_mix(h ^ key);
    h = mark_mix(h ^ stage);
    return(static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0));
  }

  // the seed handed to the decoder for a mark, so that a drawing decoder's draw for it differs from
  // its draws for the locations
  inline unsigned int mark_seed(unsigned int seed, uint64_t key){
    return(static_cast<unsigned int>(mark_mix(0x7A11ULL ^ (static_cast<uint64_t>(seed) << 32) ^ key)));
  }

  inline int mark_pick(double u, int n){
    int i = static_cast<int>(u * n);
    return(i < n ? i : n - 1);
  }

  // keys of the shared draws: one per empty cell (estimate), held-out datum (leave-one-out) or
  // cell and fold (k-fold)
  inline uint64_t mark_key_estimate(int cell){ return((static_cast<uint64_t>(cell) << 4) | 1ULL); }
  inline uint64_t mark_key_loo(int datum){ return((static_cast<uint64_t>(datum) << 4) | 2ULL); }
  inline uint64_t mark_key_kfold(int cell, int fold){
    return((((static_cast<uint64_t>(cell) << 16) | static_cast<uint64_t>(fold & 0xFFFF)) << 4) | 3ULL);
  }

  // one empty cell (or held-out datum) to fill: the cell itself (never a candidate, -1 if none),
  // the point the source cell predicts at, the key of its draw and the result rows it fills
  struct MarkTarget {
    int cell;
    std::vector<float> point;
    uint64_t key;
    std::vector<int> rows;
  };
}

#endif
