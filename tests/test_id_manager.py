import pytest
from tupimage.id_manager import *
from itertools import islice
from datetime import datetime, timedelta


@pytest.mark.parametrize("fixed_bits", range(7))
def test_id_subspace_init_correct(fixed_bits):
    """Test that IDSubspace is initialized correctly."""
    for val in range(1 << fixed_bits):
        subsp = IDSubspace(fixed_bits, val)
        assert subsp.fixed_bits == fixed_bits
        assert subsp.value == val


@pytest.mark.parametrize("fixed_bits", range(7))
def test_id_subspace_init_value_oob(fixed_bits):
    """Test that IDSubspace raises ValueError when initialized with an
    out-of-bounds value (negative or requiring more than `fixed_bits` bits)."""
    for val in [-1, 1 << fixed_bits]:
        with pytest.raises(ValueError):
            IDSubspace(fixed_bits, val)


@pytest.mark.parametrize("fixed_bits", [-1, 7, 8, 16, 32])
def test_id_subspace_init_incorrect_fixed_bits(fixed_bits):
    """Test that IDSubspace raises ValueError when initialized with an
    incorrect number of fixed bits."""
    with pytest.raises(ValueError):
        IDSubspace(fixed_bits, 0)


@pytest.mark.parametrize("fixed_bits", range(7))
def test_id_subspace_all_bytes(fixed_bits):
    # Test for several values of fixed bits.
    for val in [0, 1, (1 << fixed_bits) - 1, (1 << fixed_bits) // 2]:
        if val >= 1 and fixed_bits == 0:
            continue
        subsp = IDSubspace(fixed_bits, val)
        all_bytes = list(subsp._all_bytes())
        # Check length, sortedness, uniqueness.
        assert len(all_bytes) == 1 << (8 - fixed_bits)
        assert sorted(all_bytes) == all_bytes
        assert len(all_bytes) == len(set(all_bytes))
        # Check non-zero bytes too.
        all_nz_bytes = list(subsp._all_nonzero_bytes())
        assert 0 not in all_nz_bytes
        if len(all_bytes) != len(all_nz_bytes):
            # If there is a zero byte, it must be the first one.
            assert all_bytes[0] == 0
            assert all_nz_bytes == all_bytes[1:]
        else:
            # Otherwise, non-zero bytes are the same as all bytes.
            assert all_nz_bytes == all_bytes
        # Check that all bytes have the correct value of lower bits.
        for b in all_bytes:
            assert b & subsp._mask() == val


@pytest.mark.parametrize("fixed_bits", range(7))
def test_id_subspace_rand_nonzero_byte(fixed_bits):
    """Test that IDSubspace._rand_nonzero_byte() generates a non-zero byte with
    some bits fixed."""
    for val in [0, 1, (1 << fixed_bits) - 1, (1 << fixed_bits) // 2]:
        if val >= 1 and fixed_bits == 0:
            continue
        subsp = IDSubspace(fixed_bits, val)
        # The set of all non-zero bytes.
        all_nz_bytes = set(subsp._all_nonzero_bytes())
        # Generate many random bytes.
        for _ in range(10000):
            b = subsp._rand_nonzero_byte()
            # Check the fixed bits and non-zero-ness.
            assert b & subsp._mask() == val
            assert b != 0
            # Remove the generated byte from the set.
            all_nz_bytes.discard(b)
        # Check that we have generated all non-zero bytes.
        assert not all_nz_bytes


def test_id_features_init():
    # These are all the possible values of IDFeatures.
    all_vals = {
        IDFeatures(0),
        IDFeatures(8),
        IDFeatures(24),
        IDFeatures(8, False),
        IDFeatures(24, False),
    }
    assert all_vals == set(IDFeatures.all_values())
    assert len(set(idf.namespace_name() for idf in all_vals)) == len(all_vals)


@pytest.mark.parametrize("id_features", IDFeatures.all_values())
@pytest.mark.parametrize(
    "subspace",
    [
        IDSubspace(),
        IDSubspace(1, 0),
        IDSubspace(1, 1),
        IDSubspace(6, 0),
        IDSubspace(6, 1 << 5),
        IDSubspace(5, 0),
        IDSubspace(4, 0),
    ],
)
def test_id_features_all_ids(id_features: IDFeatures, subspace: IDSubspace):
    """Partially test the correctness of IDFeatures.all_ids().""" ""
    ids = []
    # We check only some prefix of all_ids.
    for id in islice(id_features.all_ids(subspace), 10000):
        ids.append(id)
        # Check basics.
        assert id > 0
        assert id.bit_length() <= 32
        # Check that it's in the correct id feature space.
        assert id_features.contains(id)
        assert IDFeatures.from_id(id) == id_features
        # Check that it's in the correct subspace using different methods.
        assert id & id_features.subspace_mask(
            subspace
        ) == id_features.subspace_masked_value(subspace)
        assert id_features.contains_and_in_subspace(id, subspace)
        # Check that it's not in any other id feature space.
        for other in IDFeatures.all_values():
            if other != id_features:
                assert not other.contains(id)
        # Check that certain bits are zero or non-zero.
        if id_features.use_3rd_diacritic:
            assert id & 0xFF000000 != 0
        else:
            assert id & 0xFF000000 == 0
        if id_features.color_bits == 0:
            assert id & 0x00FFFFFF == 0
        if id_features.color_bits == 8:
            assert id & 0x00FFFF00 == 0
            assert id & 0x000000FF != 0
        if id_features.color_bits == 24:
            assert id & 0x00FFFF00 != 0
    assert len(ids) <= id_features.subspace_size(subspace)
    assert len(ids) == len(set(ids))
    # If the subspace is small enough, we expect all ids to be generated.
    if id_features.subspace_size(subspace) < 10000:
        assert len(ids) == id_features.subspace_size(subspace)


@pytest.mark.parametrize("id_features", IDFeatures.all_values())
@pytest.mark.parametrize(
    "subspace",
    [
        IDSubspace(),
        IDSubspace(1, 0),
        IDSubspace(1, 1),
        IDSubspace(6, 0),
        IDSubspace(6, 1 << 5),
        IDSubspace(5, 0),
        IDSubspace(4, 0),
    ],
)
def test_id_features_gen_random_id(
    id_features: IDFeatures, subspace: IDSubspace
):
    """Test random generation of ids in a subspace."""
    ids = set()
    for i in range(10000):
        id = id_features.gen_random_id(subspace)
        ids.add(id)
        # Check basics.
        assert id > 0
        assert id.bit_length() <= 32
        # Check that it's in the correct id feature space and subspace.
        assert id_features.contains(id)
        assert id_features.contains_and_in_subspace(id, subspace)
        assert id & id_features.subspace_mask(
            subspace
        ) == id_features.subspace_masked_value(subspace)
        assert IDFeatures.from_id(id) == id_features
        # Check that it's not in any other id feature space.
        for other in IDFeatures.all_values():
            if other != id_features:
                assert not other.contains(id)
        # Check that certain bits are zero or non-zero.
        if id_features.use_3rd_diacritic:
            assert id & 0xFF000000 != 0
        else:
            assert id & 0xFF000000 == 0
        if id_features.color_bits == 0:
            assert id & 0x00FFFFFF == 0
        if id_features.color_bits == 8:
            assert id & 0x00FFFF00 == 0
            assert id & 0x000000FF != 0
        if id_features.color_bits == 24:
            assert id & 0x00FFFF00 != 0
    assert len(ids) <= id_features.subspace_size(subspace)
    # If the subspace is small enough, we expect all ids to be generated.
    if id_features.subspace_size(subspace) < 1000:
        assert len(ids) == id_features.subspace_size(subspace)


def test_id_manager_single_id():
    """Generate a single id for some subspaces in each id-feature space.
    Subspaces may intersect."""
    subspaces = [
        IDSubspace(),
        IDSubspace(1, 0),
        IDSubspace(1, 1),
        IDSubspace(6, 0),
        IDSubspace(6, 1 << 5),
        IDSubspace(5, 0),
        IDSubspace(4, 0),
    ]
    idman = IDManager(":memory:")
    for id_features in IDFeatures.all_values():
        for subspace in subspaces:
            # The path is just the space and subspace names.
            path = str(id_features) + " " + str(subspace)
            mtime = datetime.now()
            # Get an id and check that it's correct.
            id = idman.get_id(
                path=path,
                mtime=mtime,
                id_features=id_features,
                subspace=subspace,
            )
            assert id_features.contains_and_in_subspace(id, subspace)
            # Check that we get the same id if we call get_id() again.
            id2 = idman.get_id(
                path=path,
                mtime=mtime,
                id_features=id_features,
                subspace=subspace,
            )
            assert id == id2
            # Check get_info().
            info = idman.get_info(id)
            assert info.id == id
            assert info.path == path
            assert info.mtime == mtime
            assert abs(info.atime - datetime.now()) < timedelta(milliseconds=2)
            # Subspaces may intersect, so we check that there is no other id in
            # the same subspace only if it's large enough.
            if id_features.subspace_size(subspace) >= 16:
                another_id = id
                while id == another_id:
                    another_id = id_features.gen_random_id(subspace)
                assert idman.get_info(another_id) is None
                # Check that set_id() works for non-existing IDs.
                idman.set_id(another_id, path="another", mtime=mtime)
                assert idman.get_info(another_id).path == "another"
                # Check that del_id() works.
                idman.del_id(another_id)
                assert idman.get_info(another_id) is None
                assert idman.get_info(id) is not None
            # Check that set_id() works for existing IDs.
            idman.set_id(id, path="new", mtime=mtime)
            assert idman.get_info(id).path == "new"


@pytest.mark.parametrize("id_features", IDFeatures.all_values())
@pytest.mark.parametrize("fixed_bits", [0, 1, 4, 5, 6])
def test_id_manager_disjoint_subspaces(id_features: IDFeatures, fixed_bits):
    """Generate many ids for disjoint subspaces in each id-feature space."""
    subspaces = [
        IDSubspace(fixed_bits, v) for v in range(min(10, 1 << fixed_bits))
    ]
    # We will have 1100 distinct "image" paths, so some of the paths will be
    # repeated 2 or 3 times.
    num_distinct_paths = 1100
    idman = IDManager(":memory:")
    mtime = datetime.now()
    subspace_to_paths = {}
    for i in range(128):
        for subspace in subspaces:
            paths = subspace_to_paths.setdefault(subspace, [])
            for j in range(20):
                path = str((i * 20 + j) % num_distinct_paths)
                paths.append(path)
                # Get an id and check its basic correctness.
                id = idman.get_id(
                    path=path,
                    mtime=mtime,
                    id_features=id_features,
                    subspace=subspace,
                )
                assert id_features.contains_and_in_subspace(id, subspace)
                info = idman.get_info(id)
                assert info.id == id
                assert info.path == path
                assert info.mtime == mtime
                assert abs(info.atime - datetime.now()) < timedelta(
                    milliseconds=5
                )
    # Now check the IDs stored in the database.
    for subspace in subspaces:
        subspace_size = id_features.subspace_size(subspace)
        stored_paths = [
            info.path for info in idman.get_all(id_features, subspace)
        ]
        stored_paths.reverse()
        original_paths = subspace_to_paths[subspace]
        assert len(stored_paths) <= num_distinct_paths
        if subspace_size >= num_distinct_paths * 3:
            # If the subspace is large enough, we expect all paths to be stored.
            assert len(stored_paths) == num_distinct_paths
        elif subspace_size >= num_distinct_paths:
            # If it's not very large, some cleanups may be required, resulting
            # in some paths being removed.
            assert len(stored_paths) >= num_distinct_paths * 0.75
        if subspace_size < min(num_distinct_paths, 256):
            # If the subspace is small, we will not do cleanups, and will just
            # reclaim the oldest ids instead.
            assert len(stored_paths) == subspace_size
        # Finally, the stored paths must be the last ones that were added.
        assert stored_paths == original_paths[-len(stored_paths) :]
