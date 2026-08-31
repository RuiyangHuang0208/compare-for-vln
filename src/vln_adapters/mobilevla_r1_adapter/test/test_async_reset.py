from mobilevla_r1_adapter.contracts import RequestIdentity, response_is_current


def test_episode_generation_and_request_id_all_must_match():
    identity = RequestIdentity("episode-2", 7, "request-9")
    good = {"episode_id": "episode-2", "generation": 7, "request_id": "request-9"}
    assert response_is_current(good, identity)
    for field, value in (("episode_id", "episode-1"), ("generation", 6), ("request_id", "late")):
        stale = dict(good)
        stale[field] = value
        assert not response_is_current(stale, identity)

