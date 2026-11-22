from training.train_tunisl10 import TO_ANSWER, holdout_split, passes_gate


def test_gate():
    ok = {"oui": False, "non": False}
    assert passes_gate(0.90, ok) is True
    assert passes_gate(0.89, ok) is False
    assert passes_gate(0.99, {"oui": True, "non": False}) is False


def test_holdout_is_the_last_block_when_a_sign_has_two_takes():
    train, held, weak = holdout_split(30, [[0, 17], [18, 29]])
    assert list(held) == list(range(18, 30)) and list(train) == list(range(18)) and weak is False


def test_holdout_falls_back_to_the_last_quarter_and_says_it_is_weak():
    train, held, weak = holdout_split(20, [[0, 19]])
    assert list(held) == [15, 16, 17, 18, 19] and len(train) == 15 and weak is True


def test_every_tunisian_sign_maps_to_an_answer_the_flow_knows():
    from consultation.flow import Flow

    since = set(Flow.load().question("since").options)
    assert TO_ANSWER["oui"] is True and TO_ANSWER["non"] is False
    assert {v for v in TO_ANSWER.values() if isinstance(v, str)} <= since and len(TO_ANSWER) == 10
