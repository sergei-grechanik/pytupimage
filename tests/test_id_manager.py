import random
from datetime import datetime, timedelta
from itertools import islice

import pytest

from tupimage.id_manager import *


def interesting_subspaces() -> Iterator[IDSubspace]:
    yield IDSubspace(0, 256)
    yield IDSubspace(1, 256)
    yield IDSubspace(0, 255)
    yield IDSubspace(1, 255)
    yield IDSubspace(255, 256)
    for d in [1, 2, 3, 32, 113, 128]:
        for i in [0, 1, 2, 3, 32, 64, 100, 113, 128, 250, 254]:
            if i + d > 256:
                break
            if i + d != 1:
                yield IDSubspace(i, i + d)


def several_interesting_subspaces() -> Iterator[IDSubspace]:
    yield IDSubspace(0, 256)
    yield IDSubspace(1, 256)
    yield IDSubspace(0, 255)
    yield IDSubspace(1, 255)
    yield IDSubspace(255, 256)
    yield IDSubspace(0, 2)
    yield IDSubspace(0, 3)
    yield IDSubspace(0, 64)
    yield IDSubspace(0, 113)
    yield IDSubspace(1, 2)
    yield IDSubspace(1, 3)
    yield IDSubspace(1, 64)
    yield IDSubspace(1, 113)


@pytest.mark.parametrize("begin", range(256))
def test_id_subspace_init_correct(begin: int):
    """Test that IDSubspace is initialized correctly."""
    for last in range(begin, 256):
        if last == 0:
            # IDSubspace(0, 1) is not allowed.
            continue
        end = last + 1
        subsp = IDSubspace(begin, end)
        assert subsp.begin == begin
        assert subsp.end == end
        assert subsp.from_string(str(subsp)) == subsp


def test_id_subspace_from_string():
    assert IDSubspace.from_string("") == IDSubspace(0, 256)
    assert IDSubspace.from_string("") == IDSubspace()
    assert IDSubspace.from_string("0:2") == IDSubspace(0, 2)
    assert IDSubspace.from_string("0:256") == IDSubspace(0, 256)
    assert IDSubspace.from_string("1:256") == IDSubspace(1, 256)
    assert IDSubspace.from_string("100:200") == IDSubspace(100, 200)
    assert IDSubspace.from_string("255:256") == IDSubspace(255, 256)


@pytest.mark.parametrize(
    "pair",
    [
        (0, 1),
        (-1, 10),
        (10, 257),
        (0, 0),
        (10, 10),
        (10, 9),
        (256, 256),
        (256, 0),
        (10, -1),
    ],
)
def test_id_subspace_incorrect(pair):
    with pytest.raises(ValueError):
        IDSubspace(pair[0], pair[1])


@pytest.mark.parametrize("subsp", interesting_subspaces())
def test_id_subspace_all_bytes(subsp: IDSubspace):
    all_bytes = list(subsp.all_byte_values())
    # Check length, sortedness, uniqueness.
    assert len(all_bytes) == subsp.num_byte_values()
    assert sorted(all_bytes) == all_bytes
    assert len(all_bytes) == len(set(all_bytes))
    # Check non-zero bytes too.
    all_nz_bytes = list(subsp.all_nonzero_byte_values())
    assert len(all_nz_bytes) == subsp.num_nonzero_byte_values()
    assert 0 not in all_nz_bytes
    if len(all_bytes) != len(all_nz_bytes):
        # If there is a zero byte, it must be the first one.
        assert all_bytes[0] == 0
        assert all_nz_bytes == all_bytes[1:]
    else:
        # Otherwise, non-zero bytes are the same as all bytes.
        assert all_nz_bytes == all_bytes
    # Check `contains_byte`.
    all_bytes = set(all_bytes)
    for b in range(256):
        assert (b in all_bytes) == subsp.contains_byte(b)


@pytest.mark.parametrize("subsp", several_interesting_subspaces())
def test_id_subspace_rand_nonzero_byte(subsp: IDSubspace):
    """Test that IDSubspace.rand_nonzero_byte() generates a non-zero byte from the
    range."""
    # The set of all non-zero bytes.
    all_nz_bytes = set(subsp.all_nonzero_byte_values())
    # Generate many random bytes.
    for _ in range(10000):
        b = subsp.rand_nonzero_byte()
        assert subsp.contains_byte(b)
        # Remove the generated byte from the set.
        all_nz_bytes.discard(b)
    # Check that we have generated all non-zero bytes.
    assert not all_nz_bytes


@pytest.mark.parametrize("subsp", interesting_subspaces())
def test_subspace_split(subsp: IDSubspace):
    for i in range(1, subsp.num_nonzero_byte_values()):
        subs = subsp.split(i)
        assert len(subs) == i
        assert sum(sub.num_byte_values() for sub in subs) == subsp.num_byte_values()
        assert (
            sum(sub.num_nonzero_byte_values() for sub in subs)
            == subsp.num_nonzero_byte_values()
        )
        assert subs[0].begin == subsp.begin
        assert subs[-1].end == subsp.end
        for j in range(i - 1):
            assert subs[j].end == subs[j + 1].begin


def test_id_space_init():
    # These are all the possible values of IDSpace.
    all_vals = {
        IDSpace(0),
        IDSpace(8),
        IDSpace(24),
        IDSpace(8, False),
        IDSpace(24, False),
    }
    assert all_vals == set(IDSpace.all_values())
    assert len(set(idf.namespace_name() for idf in all_vals)) == len(all_vals)


@pytest.mark.parametrize("id_space", IDSpace.all_values())
@pytest.mark.parametrize(
    "subspace",
    several_interesting_subspaces(),
)
def test_id_space_all_ids(id_space: IDSpace, subspace: IDSubspace):
    """Partially test the correctness of IDSpace.all_ids()."""
    ids = []
    # We check only some prefix of all_ids.
    for id in islice(id_space.all_ids(subspace), 10000):
        ids.append(id)
        # Check basics.
        assert id > 0
        assert id.bit_length() <= 32
        # Check that it's in the correct id space.
        assert id_space.contains(id)
        assert IDSpace.from_id(id) == id_space
        # Check that it's in the correct subspace using different methods.
        begin, end = id_space.subspace_masked_range(subspace)
        assert begin <= (id & id_space.subspace_byte_mask()) < end
        assert id_space.contains_and_in_subspace(id, subspace)
        # Check that it's not in any other id space.
        for other in IDSpace.all_values():
            if other != id_space:
                assert not other.contains(id)
        # Check that certain bits are zero or non-zero.
        if id_space.use_3rd_diacritic:
            assert id & 0xFF000000 != 0
        else:
            assert id & 0xFF000000 == 0
        if id_space.color_bits == 0:
            assert id & 0x00FFFFFF == 0
        if id_space.color_bits == 8:
            assert id & 0x00FFFF00 == 0
            assert id & 0x000000FF != 0
        if id_space.color_bits == 24:
            assert id & 0x00FFFF00 != 0
    assert len(ids) <= id_space.subspace_size(subspace)
    assert len(ids) == len(set(ids))
    # If the subspace is small enough, we expect all ids to be generated.
    if id_space.subspace_size(subspace) < 10000:
        assert len(ids) == id_space.subspace_size(subspace)


@pytest.mark.parametrize("id_space", IDSpace.all_values())
@pytest.mark.parametrize(
    "subspace",
    several_interesting_subspaces(),
)
def test_id_space_gen_random_id(id_space: IDSpace, subspace: IDSubspace):
    """Test random generation of ids in a subspace."""
    ids = set()
    for i in range(10000):
        id = id_space.gen_random_id(subspace)
        ids.add(id)
        # Check basics.
        assert id > 0
        assert id.bit_length() <= 32
        # Check that it's in the correct id space and subspace.
        assert id_space.contains(id)
        assert id_space.contains_and_in_subspace(id, subspace)
        begin, end = id_space.subspace_masked_range(subspace)
        assert begin <= (id & id_space.subspace_byte_mask()) < end
        assert IDSpace.from_id(id) == id_space
        # Check that it's not in any other id space.
        for other in IDSpace.all_values():
            if other != id_space:
                assert not other.contains(id)
        # Check that certain bits are zero or non-zero.
        if id_space.use_3rd_diacritic:
            assert id & 0xFF000000 != 0
        else:
            assert id & 0xFF000000 == 0
        if id_space.color_bits == 0:
            assert id & 0x00FFFFFF == 0
        if id_space.color_bits == 8:
            assert id & 0x00FFFF00 == 0
            assert id & 0x000000FF != 0
        if id_space.color_bits == 24:
            assert id & 0x00FFFF00 != 0
    assert len(ids) <= id_space.subspace_size(subspace)
    # If the subspace is small enough, we expect all ids to be generated.
    if id_space.subspace_size(subspace) < 1000:
        assert len(ids) == id_space.subspace_size(subspace)


def test_id_manager_single_id():
    """Generate a single id for some subspaces in each id space.
    Subspaces may intersect."""
    subspaces = list(several_interesting_subspaces())
    idman = IDManager(":memory:")
    for id_space in IDSpace.all_values():
        for subspace in subspaces:
            # The description is just the space and subspace names.
            description = str(id_space) + " " + str(subspace)
            # Get an id and check that it's correct.
            id = idman.get_id(
                description=description,
                id_space=id_space,
                subspace=subspace,
            )
            assert id_space.contains_and_in_subspace(id, subspace)
            # Check that we get the same id if we call get_id() again.
            id2 = idman.get_id(
                description=description,
                id_space=id_space,
                subspace=subspace,
            )
            assert id == id2
            # Check get_info().
            info = idman.get_info(id)
            assert info
            assert info.id == id
            assert info.description == description
            assert info.atime is not None
            assert abs(info.atime - datetime.now()) < timedelta(milliseconds=20)
            # Subspaces may intersect, so we check that there is no other id in
            # the same subspace only if it's large enough.
            if id_space.subspace_size(subspace) >= 1000:
                another_id = id
                while id == another_id:
                    another_id = id_space.gen_random_id(subspace)
                assert idman.get_info(another_id) is None
                # Check that set_id() works for non-existing IDs.
                idman.set_id(another_id, description="another")
                assert idman.get_info(another_id).description == "another"
                # Check that del_id() works.
                idman.del_id(another_id)
                assert idman.get_info(another_id) is None
                assert idman.get_info(id) is not None
            # Check that set_id() works for existing IDs.
            idman.set_id(id, description="new")
            assert idman.get_info(id).description == "new"


@pytest.mark.parametrize("id_space", IDSpace.all_values())
@pytest.mark.parametrize("big_subspace_end", [8, 14, 15, 255, 256])
@pytest.mark.parametrize("num_subspaces", [1, 2, 7])
def test_id_manager_disjoint_subspaces(
    id_space: IDSpace, big_subspace_end: int, num_subspaces: int
):
    """Generate many ids for disjoint subspaces in each id space."""
    subspaces = IDSubspace(0, big_subspace_end).split(num_subspaces)
    # We will have 1100 distinct "image" descriptions, so some of the descriptions will be
    # repeated 2 or 3 times.
    num_distinct_descriptions = 1100
    idman = IDManager(":memory:")
    subspace_to_descriptions = {}
    for i in range(128):
        for subspace in subspaces:
            descriptions = subspace_to_descriptions.setdefault(subspace, [])
            for j in range(20):
                description = str((i * 20 + j) % num_distinct_descriptions)
                descriptions.append(description)
                # Get an id and check its basic correctness.
                id = idman.get_id(
                    description=description,
                    id_space=id_space,
                    subspace=subspace,
                )
                now = datetime.now()
                assert id_space.contains_and_in_subspace(id, subspace)
                info = idman.get_info(id)
                assert info
                assert info.id == id
                assert info.description == description
                assert info.atime is not None
                assert abs(info.atime - now) < timedelta(milliseconds=20)
    # Now check the IDs stored in the database.
    for subspace in subspaces:
        subspace_size = id_space.subspace_size(subspace)
        stored_descriptions = [
            info.description for info in idman.get_all(id_space, subspace)
        ]
        stored_descriptions.reverse()
        original_descriptions = subspace_to_descriptions[subspace]
        assert len(stored_descriptions) <= num_distinct_descriptions
        if subspace_size >= num_distinct_descriptions * 3:
            # If the subspace is large enough, we expect all descriptions to be stored.
            assert len(stored_descriptions) == num_distinct_descriptions
        elif subspace_size >= num_distinct_descriptions:
            # If it's not very large, some cleanups may be required, resulting
            # in some descriptions being removed.
            assert len(stored_descriptions) >= num_distinct_descriptions * 0.75
        if subspace_size < min(num_distinct_descriptions, 256):
            # If the subspace is small, we will not do cleanups, and will just
            # reclaim the oldest ids instead.
            assert len(stored_descriptions) == subspace_size
        # Finally, the stored descriptions must be the last ones that were added.
        assert stored_descriptions == original_descriptions[-len(stored_descriptions) :]


def test_id_manager_uploads():
    """Test marking IDs as uploaded to terminals."""
    idman = IDManager(":memory:")
    terminals = ["term" + str(i) for i in range(10)]
    term_to_infos = {term: [] for term in terminals}
    for i in range(1000):
        description = str(i)
        size = random.randint(0, 100000)
        id = idman.get_id(
            description=description,
            id_space=IDSpace(),
        )
        for term in terminals:
            assert idman.needs_uploading(id, term)
            assert idman.get_upload_info(id, term) is None
            if random.random() < 0.5:
                idman.mark_uploaded(id, term, size=size)
                assert not idman.needs_uploading(id, term)
                info = idman.get_upload_info(id, term)
                assert info
                term_to_infos[term].append(info)
                assert info.upload_time is not None
                assert abs(info.upload_time - datetime.now()) < timedelta(
                    milliseconds=20
                )
                assert info.size == size
                assert info.description == description
                assert info.id == id
                assert info.terminal == term
                assert info.bytes_ago == size
                assert info.uploads_ago == 1
    # Compute bytes_ago and uploads_ago in python and compare to the results we
    # get from the database.
    for term in terminals:
        infos = term_to_infos[term]
        infos.reverse()
        for i, info in enumerate(infos):
            info.uploads_ago = i + 1
            if i >= 1:
                info.bytes_ago = infos[i - 1].bytes_ago + info.size
        for info in infos:
            assert idman.get_upload_info(info.id, term) == info


def test_id_manager_uploads_example():
    """Test marking IDs as uploaded to terminals again."""
    idman = IDManager(":memory:")
    # Generate some IDs.
    id1 = idman.get_id("1", IDSpace())
    id2 = idman.get_id("2", IDSpace())
    id3 = idman.get_id("3", IDSpace())
    id4 = idman.get_id("4", IDSpace())
    # Mark them as uploaded to term1 and term2.
    idman.mark_uploaded(id1, "term1", size=100)
    idman.mark_uploaded(id1, "term2", size=100)
    idman.mark_uploaded(id2, "term1", size=200)
    idman.mark_uploaded(id2, "term2", size=200)
    idman.mark_uploaded(id3, "term1", size=300)
    idman.mark_uploaded(id3, "term2", size=300)
    idman.mark_uploaded(id4, "term1", size=400)
    idman.mark_uploaded(id4, "term2", size=400)
    # Check info.
    assert idman.get_upload_info(id1, "term1").bytes_ago == 1000
    assert idman.get_upload_info(id1, "term1").uploads_ago == 4
    assert idman.get_upload_info(id1, "term2").bytes_ago == 1000
    assert idman.get_upload_info(id1, "term2").uploads_ago == 4
    # Manually set id1 to be a different image.
    idman.set_id(id1, "1-re")
    # Now id1 needs uploading.
    assert idman.needs_uploading(id1, "term1")
    # Mark it as uploaded again to term1, but not to term2.
    idman.mark_uploaded(id1, "term1", size=100)
    # Now it doesn't need uploading to term1 but still needs uploading to term2.
    assert not idman.needs_uploading(id1, "term1")
    assert idman.needs_uploading(id1, "term2")
    # Check the new info.
    assert idman.get_upload_info(id1, "term1").bytes_ago == 100
    assert idman.get_upload_info(id1, "term1").uploads_ago == 1
    assert idman.get_upload_info(id1, "term2").bytes_ago == 1000
    assert idman.get_upload_info(id1, "term2").uploads_ago == 4
    assert idman.get_upload_info(id2, "term1").bytes_ago == 1000
    assert idman.get_upload_info(id2, "term1").uploads_ago == 4
    assert idman.get_upload_info(id2, "term2").bytes_ago == 900
    assert idman.get_upload_info(id2, "term2").uploads_ago == 3
    # Do a cleanup.
    idman.cleanup_uploads(max_uploads=3)
    assert idman.get_upload_info(id1, "term1") is not None
    assert idman.get_upload_info(id4, "term2") is not None
    assert idman.get_upload_info(id4, "term1") is not None
    assert idman.get_upload_info(id3, "term1") is None
    assert idman.get_upload_info(id3, "term2") is None
    assert idman.get_upload_info(id2, "term1") is None
    assert idman.get_upload_info(id2, "term1") is None
    assert idman.get_upload_info(id1, "term2") is None
