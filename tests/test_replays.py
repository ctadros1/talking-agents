from emcomm.replays import pick_episodes


def test_pick_one_success_per_target_plus_failures():
    records = [{"episode": i, "success": i % 3 != 0, "target": i % 4} for i in range(24)]
    chosen = pick_episodes(records, n_targets=4)
    successes = [records[i] for i in chosen if records[i]["success"]]
    failures = [records[i] for i in chosen if not records[i]["success"]]
    assert sorted({r["target"] for r in successes}) == [0, 1, 2, 3] and len(successes) == 4
    assert len(failures) == 2 and chosen == sorted(chosen)
