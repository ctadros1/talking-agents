from emcomm.sweep import parse_seeds, plan_jobs, run_dir


def test_parse_seeds():
    assert parse_seeds("0-3") == [0, 1, 2, 3]
    assert parse_seeds("1000-1002,7") == [1000, 1001, 1002, 7]


def test_plan_is_seed_major_and_skips_finished(tmp_path):
    done = run_dir(tmp_path, "comm", 0)
    done.mkdir(parents=True)
    (done / "final.pt").write_bytes(b"x")
    jobs = plan_jobs(tmp_path, ["comm", "no_comm"], [0, 1])
    assert jobs == [("no_comm", 0), ("comm", 1), ("no_comm", 1)]
