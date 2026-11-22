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


def test_frames_are_grouped_by_signer_from_their_colours():
    import numpy as np

    from training.train_tunisl10 import cluster_signers

    rng = np.random.default_rng(0)
    centres = np.array([[30, 20, 20], [220, 225, 225], [60, 70, 85]], dtype=float)   # black, white, grey-blue tops
    truth = rng.integers(0, 3, 120)
    signatures = centres[truth] + rng.normal(0, 6, (120, 3))
    groups = cluster_signers(signatures)
    assert len(set(groups)) == 3
    for person in range(3):   # everyone of one true signer shares one group
        assert len(set(groups[truth == person])) == 1


def test_one_signer_is_held_out_of_every_sign_they_appear_in():
    import numpy as np

    from training.train_tunisl10 import choose_held_out_signer

    signers = {"oui": np.array([0, 0, 0, 1, 1, 2, 2, 2, 2]), "non": np.array([0, 0, 1, 1, 1, 2]),
               "lyoum": np.array([0, 0, 0, 1, 1])}
    # signer 1 is in every sign and is the smaller of the two who appear in both oui and non
    assert choose_held_out_signer(signers) == 1
