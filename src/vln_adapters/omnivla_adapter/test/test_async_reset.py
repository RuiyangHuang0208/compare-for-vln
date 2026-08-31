from omnivla_adapter.contracts import response_is_current


def response(episode="e1", generation=4, request="r1"):
    return {
        "episode_id": episode,
        "generation": generation,
        "request_id": request,
        "modality_id": 7,
    }


def test_current_response_identity():
    assert response_is_current(response(), "e1", 4, "r1")


def test_reset_discards_late_response():
    assert not response_is_current(response(generation=3), "e1", 4, "r1")
    assert not response_is_current(response(episode="old"), "e1", 4, "r1")
    assert not response_is_current(response(request="old"), "e1", 4, "r1")


def test_wrong_modality_is_stale_or_invalid():
    value = response()
    value["modality_id"] = 6
    assert not response_is_current(value, "e1", 4, "r1")

