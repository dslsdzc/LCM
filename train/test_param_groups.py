"""Parameter groups (R3): trainable / EMA / frozen must round-trip exactly.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_param_groups.py -v
"""
import numpy as np
import jax.numpy as jnp

from train.param_groups import (
    EMA_PATHS, FROZEN_PATHS, merge_params, partition_params,
)


def test_ema_paths_are_the_frozen_contract():
    """R3/Ruling 4: exactly these EMA paths, danger is the only frozen param.

    gvalue is a runtime state object (``state['gvalue']``), never a member of
    ``params``, so it must not appear here.
    """
    assert EMA_PATHS == {
        ('sparse', 'C'), ('manifold', 'C'),
        ('binding', 'key_cb'), ('binding', 'val_cb'), ('binding', 'bind_cb'),
        ('self', 'C_self'),
    }
    assert FROZEN_PATHS == {('danger', 'C')}


def test_partition_roundtrip():
    p = {'encoder': {'w': jnp.ones((2, 2))},
         'sparse': {'C': jnp.ones((4, 2)), 'zero_vec': jnp.zeros((1, 2))},
         'manifold': {'C': jnp.ones((4, 2)), 'T': jnp.ones((4, 2, 1))},
         'binding': {'key_cb': [{'A': jnp.ones((4, 2))}], 'A_k': jnp.ones((2, 2))},
         'danger': {'C': jnp.ones((4, 2))}}
    tr, em, fr = partition_params(p)
    assert 'sparse' not in tr
    assert 'sparse' in em
    assert 'danger' in fr
    assert 'binding' in tr and 'key_cb' not in tr['binding']
    assert 'A_k' in tr['binding']
    assert merge_params(tr, em, fr).keys() == p.keys()


def _leaf_paths(tree, prefix=()):
    """Yield (path, value) for every non-container node, including None.

    JAX's own tree flattening drops ``None``; a checkpoint round-trip must not.
    """
    if isinstance(tree, dict):
        for key, child in tree.items():
            yield from _leaf_paths(child, prefix + (key,))
    elif isinstance(tree, (list, tuple)):
        for i, child in enumerate(tree):
            yield from _leaf_paths(child, prefix + (i,))
    else:
        yield prefix, tree


def _assert_same_value(a, b, path):
    if a is None or b is None:
        assert a is None and b is None, f"{path}: {a!r} != {b!r}"
        return
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=str(path))


def test_every_leaf_survives_and_no_leaf_is_duplicated():
    p = {
        'encoder': {'embed': jnp.arange(12.0).reshape(4, 3),
                    'layers': [{'w': jnp.ones((3, 3))}]},
        'sparse': {'C': jnp.arange(12.0).reshape(4, 3),
                   'zero_vec': jnp.zeros((1, 3))},
        'manifold': {'C': jnp.ones((4, 3)), 'T': jnp.ones((4, 3, 2))},
        'binding': {
            'A_k': jnp.ones((2, 3)),
            'key_cb': [{'A': jnp.ones((4, 3)), 'W': jnp.eye(3)},
                       {'A': jnp.zeros((4, 3)), 'W': jnp.eye(3)}],
            'val_cb': [{'A': jnp.ones((4, 3)), 'W': jnp.eye(3)}],
            'bind_cb': [{'A': jnp.ones((4, 3)), 'W': jnp.eye(3)}],
        },
        'self': {'C_self': jnp.ones((4, 3)), 'bias_self': jnp.array(0.05)},
        'danger': {'C': jnp.ones((4, 3))},
        'qwen': None,
        'z_proj': None,
    }
    tr, em, fr = partition_params(p)
    merged = merge_params(tr, em, fr)

    original = dict(_leaf_paths(p))
    rebuilt = dict(_leaf_paths(merged))
    assert set(original) == set(rebuilt), (
        f"lost={set(original) - set(rebuilt)}, "
        f"added={set(rebuilt) - set(original)}")
    for path in original:
        _assert_same_value(original[path], rebuilt[path], path)

    group_paths = ([path for path, _ in _leaf_paths(tr)]
                   + [path for path, _ in _leaf_paths(em)]
                   + [path for path, _ in _leaf_paths(fr)])
    assert len(group_paths) == len(set(group_paths)), \
        f"leaf assigned to more than one group: {group_paths}"
    assert set(group_paths) == set(original), "a leaf is in no group at all"


def test_binding_lists_are_whole_groups_and_keep_their_length():
    """binding.*_cb are lists of dicts: a list may not be split across groups."""
    p = {
        'encoder': {'w': jnp.ones((2, 2))},
        'binding': {
            'A_k': jnp.ones((2, 2)),
            'key_cb': [{'A': jnp.ones((4, 2)), 'W': jnp.eye(2)}],
            'val_cb': [{'A': jnp.ones((4, 2)), 'W': jnp.eye(2)}],
            'bind_cb': [{'A': jnp.ones((4, 2)), 'W': jnp.eye(2)}],
        },
    }
    tr, em, fr = partition_params(p)
    assert 'binding' not in fr
    for key in ('key_cb', 'val_cb', 'bind_cb'):
        assert key in em['binding']
        assert len(em['binding'][key]) == len(p['binding'][key])
        assert key not in tr['binding']
    assert 'A_k' not in em['binding'] and 'A_k' in tr['binding']
    out = merge_params(tr, em, fr)
    assert set(out['binding']) == {'A_k', 'key_cb', 'val_cb', 'bind_cb'}


def test_frozen_constants_stay_out_of_the_optimizer_tree():
    p = {
        'encoder': {'w': jnp.ones((2, 2))},
        'sparse': {'C': jnp.ones((4, 2)), 'zero_vec': jnp.zeros((1, 2))},
        # R15: C_self is one EMA-owned array; ema_update_self freezes row 0.
        'self': {'C_self': jnp.ones((4, 2)), 'bias_self': jnp.array(0.05)},
        'danger': {'C': jnp.ones((4, 2))},
        'qwen': {'model.embed_tokens.weight': jnp.ones((3, 2))},
        'z_proj': None,
    }
    tr, em, fr = partition_params(p)
    assert 'sparse' not in tr
    assert 'zero_vec' not in em['sparse']
    assert 'zero_vec' in fr['sparse']
    assert 'danger' in fr and 'danger' not in tr
    # C_self belongs wholly to the EMA group (a JAX array cannot be split), so
    # the core row is protected inside ema_update_self, not by a frozen path.
    assert 'C_self' in em['self'] and 'C_self' not in tr.get('self', {})
    # Ruling 34: bias_self is the frozen cfg.alpha_self mirror, not trainable.
    assert 'bias_self' in fr['self'] and 'bias_self' not in tr.get('self', {})
    assert 'bias_self' not in em['self']
    # The frozen Qwen bridge is ~2 GB and must never enter AdamW.
    assert 'qwen' in fr and 'qwen' not in tr
    assert 'z_proj' not in tr and 'z_proj' in fr
