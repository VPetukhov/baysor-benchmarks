"""Classification of parallel regions and waiting for OpenMP builds and for
builds with Baysor's own thread pool (cgparse, summarize.short_name,
gperf.classify)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cgparse  # noqa: E402
import gperf  # noqa: E402
import summarize  # noqa: E402

CHUNK_FOR_EACH = ("std::_Function_handler<void (long, long, int), baysor::ParallelRegion::for_each<"
                  "baysor::apply_phase<2>(baysor::ParallelRegion&, baysor::BmmData<2>&)::{lambda(int)#2}>"
                  "(long, long, long, baysor::apply_phase<2>(baysor::ParallelRegion&, baysor::BmmData<2>&)"
                  "::{lambda(int)#2}&&, baysor::Scheduling)::{lambda(long, long, int)#1}>::_M_invoke("
                  "std::_Any_data const&, long&&, long&&, int&&)")
CHUNK_DIRECT = ("std::_Function_handler<void (long, long, int), baysor::estep_phase<2>(baysor::ParallelRegion&, "
                "baysor::BmmData<2>&, bool, unsigned long)::{lambda(long, long, int)#1}>::_M_invoke("
                "std::_Any_data const&, long&&, long&&, int&&)")
CHUNK_ANON = ("std::_Function_handler<void (long, long, int), void baysor::parallel_for<baysor::(anonymous "
              "namespace)::normalize_columns_l2(Eigen::Matrix<float, -1, -1, 0, -1, -1> const&)::{lambda(int)#1}>"
              "(long, long, long, baysor::(anonymous namespace)::normalize_columns_l2(Eigen::Matrix<float, -1, -1, "
              "0, -1, -1> const&)::{lambda(int)#1}&&, baysor::Scheduling)::{lambda(long, long, int)#1}>::_M_invoke("
              "std::_Any_data const&, long&&, long&&, int&&)")
BODY = ("std::_Function_handler<void (baysor::ParallelRegion&), void baysor::bmm<2>(baysor::BmmData<2>&, int, "
        "int)::{lambda(baysor::ParallelRegion&)#1}>::_M_invoke(std::_Any_data const&, baysor::ParallelRegion&)")
OMP = "baysor::EstepStats baysor::expect_dirichlet_spatial<2>(baysor::BmmData<2>&, bool) [clone ._omp_fn.0]"


def test_region_predicates():
    assert cgparse.is_pool_chunk(CHUNK_FOR_EACH) and cgparse.is_region_fn(CHUNK_FOR_EACH)
    assert cgparse.is_omp_fn(OMP) and cgparse.is_region_fn(OMP) and not cgparse.is_pool_chunk(OMP)
    assert not cgparse.is_region_fn(BODY)          # persistent-region body: not work-shared
    for f in ("baysor::run_parallel_chunks(long, long, long, baysor::Scheduling, std::function<void (long, "
              "long, int)> const&)", "baysor::ParallelRegion::barrier()",
              "baysor::(anonymous namespace)::ThreadPool::run(baysor::(anonymous namespace)::Job&)"):
        assert cgparse.is_pool_runtime(f)
    assert not cgparse.is_pool_runtime(CHUNK_DIRECT)


def test_short_names_and_parents():
    assert summarize.short_name(CHUNK_FOR_EACH) == "baysor::apply_phase::{lambda#2}.pool_chunk"
    assert summarize.short_name(CHUNK_DIRECT) == "baysor::estep_phase::{lambda#1}.pool_chunk"
    assert summarize.short_name(CHUNK_ANON) == "baysor::{anon}::normalize_columns_l2::{lambda#1}.pool_chunk"
    assert summarize.short_name(BODY) == "baysor::bmm::{lambda#1}.pool_region"
    assert summarize.pool_parent(CHUNK_FOR_EACH) == "baysor::apply_phase"
    assert summarize.pool_parent(OMP) is None
    # OpenMP names are unchanged
    assert summarize.short_name(OMP) == "baysor::expect_dirichlet_spatial._omp_fn.0"


def test_classify_samples():
    main = ["cmd_run(x)", "main"]
    phases = [{"name": "bmm_iterations", "pattern": "void baysor::bmm<*"}]
    bmm = "void baysor::bmm<2>(baysor::BmmData<2>&, int, int)"
    stacks = [
        (10, ["baysor::foo()", CHUNK_DIRECT, "baysor::run_parallel_chunks(long)", bmm] + main),   # main, parallel
        (5, ["baysor::serial_part()", bmm] + main),                                             # main, serial
        (6, [CHUNK_DIRECT, "baysor::(anonymous namespace)::ThreadPool::worker_loop(int)",
             "start_thread"]),                                                                  # worker, parallel
        (3, ["syscall", "baysor::(anonymous namespace)::ThreadPool::worker_loop(int)", "start_thread"]),  # wait
        (2, ["gomp_barrier_wait_end", "gomp_thread_start"]),                                    # OpenMP wait
    ]
    c = gperf.classify(stacks, phases, summarize.short_name)
    assert c["samples"] == 26 and c["parallel"] == 16
    assert c["wait_pool"] == 3 and c["wait_omp"] == 2
    ph = c["phases"]["bmm_iterations"]
    assert ph["samples"] == 21 and ph["parallel"] == 16      # worker samples mapped via the region
    assert c["regions"] == {"baysor::estep_phase::{lambda#1}.pool_chunk": 16}
