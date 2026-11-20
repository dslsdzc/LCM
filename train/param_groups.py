"""Parameter groups for the cognitive model (R3).

Three groups, one invariant: each leaf belongs to exactly one of them.

    trainable  pure gradient parameters, the only ones AdamW sees
    ema        codebooks updated by the EMA pass and never by the optimizer
    frozen     read-only tensors (danger, the frozen Qwen bridge, constants)

Why this exists
---------------
AdamW's decoupled weight decay is independent of the gradient. Leaving an EMA
codebook in the optimizer tree therefore shrinks it a little every step even
though its gradient is exactly zero, and the EMA pass is no longer the only
thing that writes it. The same argument applies to ``danger`` (frozen safety
data) and to the frozen Qwen bridge (~2 GB: it must not even allocate Adam
moment buffers).

EMA paths are whole subtrees, not leaves: ``('binding', 'key_cb')`` is a *list*
of SimVQ dicts, so splitting/merging has to handle lists of dicts, not only
dicts.

Ruling 4: gvalue lives in ``state['gvalue']``, never in ``params``, so there is
no gvalue path here. Adding one would freeze something that is not in this tree.
"""
# R3: codebooks updated only by the EMA pass.
EMA_PATHS = frozenset({
    ('sparse', 'C'),
    ('manifold', 'C'),
    ('binding', 'key_cb'),
    ('binding', 'val_cb'),
    ('binding', 'bind_cb'),
    ('self', 'C_self'),
})

# R3 / Ruling 4: frozen parameter leaves. gvalue is a runtime state object and
# is deliberately absent.
FROZEN_PATHS = frozenset({
    ('danger', 'C'),
})

# Read-only constants that are stored in the params tree for shape/API
# stability but are not parameters:
#   * sparse.zero_vec is the LFQ origin. It must stay exactly zero.
#   * the Qwen bridge is a frozen pretrained model, not a cognitive parameter.
# R15 removed self.core: the frozen identity anchor is C_self[0], a row inside
# the EMA-owned C_self array. A single JAX array cannot be split across
# parameter groups, so the freeze is enforced by
# train.ema_state.ema_update_self (it never writes row 0 and keeps its EMA
# statistics at zero) rather than by assigning the row to the frozen group.
# The brief fixes FROZEN_PATHS to danger.C only; these are structural constants,
# kept in a separate set so the R3 contract stays literally true.
CONSTANT_PATHS = frozenset({
    ('sparse', 'zero_vec'),
    ('qwen',),
})

_TRAINABLE = 'trainable'
_EMA = 'ema'
_FROZEN = 'frozen'

# Distinguishes "this group does not own the node" from a real value. Never
# escapes into a returned group tree.
_ABSENT = object()


def _under(path, prefix):
    return len(path) >= len(prefix) and path[:len(prefix)] == prefix


def _owner(path):
    """Which group owns the whole subtree rooted at ``path``, or None."""
    if path in EMA_PATHS:
        return _EMA
    if path in FROZEN_PATHS or path in CONSTANT_PATHS:
        return _FROZEN
    for prefix in CONSTANT_PATHS:
        if _under(path, prefix):
            return _FROZEN
    return None


def _prune(value, path, group):
    """Return the part of ``value`` owned by ``group``, or ``_ABSENT``."""
    owner = _owner(path)
    if owner is not None:
        return value if owner == group else _ABSENT

    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            pruned = _prune(child, path + (key,), group)
            if pruned is not _ABSENT:
                out[key] = pruned
        return out if out else _ABSENT

    if isinstance(value, (list, tuple)):
        # A list is all-or-nothing unless a path above claimed it (binding's
        # codebook lists are claimed whole). Mixing groups inside one list would
        # need index placeholders in the optimizer tree, which JAX would then
        # treat as parameters; reject it instead of building that trap.
        elements = [_prune(child, path + (i,), group)
                    for i, child in enumerate(value)]
        present = [e for e in elements if e is not _ABSENT]
        if not present:
            return _ABSENT
        if len(present) != len(elements):
            raise ValueError(
                f"list at {path} mixes parameter groups; a list must belong "
                f"wholly to one group (got {len(present)}/{len(elements)})")
        return type(value)(elements)

    if value is None:
        # None is not a JAX parameter (e.g. qwen disabled, z_proj not yet
        # trained). Keep it out of the optimizer tree; merge restores it.
        return value if group == _FROZEN else _ABSENT

    return value if group == _TRAINABLE else _ABSENT


def _walk_leaves(tree, prefix=()):
    """Yield (path, value) for every non-container node, including None."""
    if isinstance(tree, dict):
        for key, child in tree.items():
            yield from _walk_leaves(child, prefix + (key,))
    elif isinstance(tree, (list, tuple)):
        for i, child in enumerate(tree):
            yield from _walk_leaves(child, prefix + (i,))
    else:
        yield prefix, tree


def partition_params(params):
    """Split ``params`` into (trainable, ema, frozen) pruned trees.

    Each returned tree contains only the leaves owned by that group. Empty
    branches are omitted, so ``'sparse' not in trainable`` when every sparse
    leaf is EMA/frozen. The three trees together contain each original leaf
    exactly once.
    """
    if not isinstance(params, dict):
        raise TypeError(f"params must be a dict, got {type(params).__name__}")

    groups = []
    for group in (_TRAINABLE, _EMA, _FROZEN):
        pruned = _prune(params, (), group)
        groups.append({} if pruned is _ABSENT else pruned)
    return tuple(groups)


def _merge_nodes(values, path):
    present = [v for v in values if v is not _ABSENT]
    if not present:
        raise ValueError(f"no group owns {path}; a leaf was lost in partition")
    if len(present) == 1:
        return present[0]
    if all(isinstance(v, dict) for v in present):
        keys = []
        for value in present:
            for key in value:
                if key not in keys:
                    keys.append(key)
        out = {}
        for key in keys:
            child = _merge_nodes(
                [v.get(key, _ABSENT) if v is not _ABSENT else _ABSENT
                 for v in values],
                path + (key,))
            out[key] = child
        return out
    raise ValueError(
        f"parameter at {path} is present in more than one group: "
        f"{[type(v).__name__ for v in present]}")


def merge_params(trainable, ema, frozen):
    """Inverse of :func:`partition_params`; rebuilds the full params dict.

    Raises if a leaf is missing or assigned twice, rather than silently
    dropping or overwriting it.
    """
    seen = {}
    for name, tree in (('trainable', trainable), ('ema', ema),
                       ('frozen', frozen)):
        for path, value in _walk_leaves(tree):
            if path in seen:
                raise ValueError(
                    f"leaf {path} appears in both {seen[path]} and {name}")
            seen[path] = name

    merged = _merge_nodes([trainable, ema, frozen], ())
    rebuilt = dict(_walk_leaves(merged))

    missing = set(seen) - set(rebuilt)
    extra = set(rebuilt) - set(seen)
    if missing or extra:
        raise ValueError(
            f"merge did not round-trip: missing={sorted(missing)}, "
            f"extra={sorted(extra)}")

    total = sum(len(list(_walk_leaves(tree)))
                for tree in (trainable, ema, frozen))
    if total != len(rebuilt):
        raise ValueError(
            f"merge duplicated or lost leaves: {total} grouped vs "
            f"{len(rebuilt)} rebuilt")
    return merged


__all__ = [
    'EMA_PATHS', 'FROZEN_PATHS', 'CONSTANT_PATHS',
    'partition_params', 'merge_params',
]
