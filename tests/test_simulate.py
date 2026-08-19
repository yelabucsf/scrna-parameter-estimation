import numpy as np

from memento import simulate


def test_sequencing_sampling_is_vectorized_and_bounded_by_input_counts():
    transcriptomes = np.array([[0, 2, 5], [3, 1, 0]])
    observed = simulate.sequencing_sampling(
        transcriptomes,
        num_reads=20,
        gen=np.random.default_rng(5),
    )

    assert observed.shape == transcriptomes.shape
    assert np.issubdtype(observed.dtype, np.integer)
    assert np.all(observed >= 0)
    assert np.all(observed <= transcriptomes)


def test_sequencing_sampling_handles_empty_input():
    transcriptomes = np.zeros((2, 3), dtype=int)
    np.testing.assert_array_equal(
        simulate.sequencing_sampling(transcriptomes, num_reads=100),
        transcriptomes,
    )
