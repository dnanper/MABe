"""
Data loading and generation functions
"""
import pandas as pd
import numpy as np
import json
import itertools
import re
from config import DROP_BODY_PARTS


def load_train_test_data(train_path, test_path):
    """Load train and test CSV files with preprocessing"""
    train = pd.read_csv(train_path)
    
    # Drop likely-sleeping MABe22 clips
    train = train.loc[~(
        train['lab_id'].astype(str).str.contains('MABe22', na=False) &
        train['mouse1_condition'].astype(str).str.lower().eq('lights on')
    )].copy()
    
    train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 
                                   'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)
    
    test = pd.read_csv(test_path)
    test['sleeping'] = (
        test['lab_id'].astype(str).str.contains('MABe22', na=False) &
        test['mouse1_condition'].astype(str).str.lower().eq('lights on')
    )
    test['n_mice'] = 4 - test[['mouse1_strain', 'mouse2_strain',
                                 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)
    
    return train, test


def generate_mouse_data(dataset, traintest, traintest_directory=None,
                        generate_single=True, generate_pair=True):
    """
    Generator that yields mouse tracking data with features and labels.
    
    Yields:
        tuple: (switch, data, meta, labels_or_actions)
            - switch: 'single' or 'pair'
            - data: DataFrame with tracking coordinates
            - meta: DataFrame with metadata
            - labels_or_actions: DataFrame (train) or list (test)
    """
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    def _to_num(x):
        if isinstance(x, (int, np.integer)):
            return int(x)
        m = re.search(r'(\d+)$', str(x))
        return int(m.group(1)) if m else None

    for _, row in dataset.iterrows():
        lab_id = row.lab_id
        video_id = row.video_id
        fps = float(row.frames_per_second)
        n_mice = int(row.n_mice)
        arena_w = float(row.get('arena_width_cm', np.nan))
        arena_h = float(row.get('arena_height_cm', np.nan))
        sleeping = bool(getattr(row, 'sleeping', False))
        arena_shape = row.get('arena_shape', 'rectangular')

        if not isinstance(row.behaviors_labeled, str):
            continue

        # Load tracking data
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@DROP_BODY_PARTS)")
        
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], 
                        index='video_frame', values=['x', 'y'])
        del vid
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid = (pvid / float(row.pix_per_cm_approx)).astype('float32', copy=False)

        # Available mouse IDs
        avail = list(pvid.columns.get_level_values('mouse_id').unique())
        avail_set = set(avail) | set(map(str, avail)) | \
                    {f"mouse{_to_num(a)}" for a in avail if _to_num(a) is not None}

        def _resolve(agent_str):
            """Return the matching mouse_id label in pvid"""
            m = re.search(r'(\d+)$', str(agent_str))
            cand = [agent_str]
            if m:
                n = int(m.group(1))
                cand = [n, n-1, str(n), f"mouse{n}", agent_str]
            for c in cand:
                if c in avail_set:
                    if c in set(avail):
                        return c
                    for a in avail:
                        if str(a) == str(c) or f"mouse{_to_num(a)}" == str(c):
                            return a
            return None

        # Parse behaviors
        vb = json.loads(row.behaviors_labeled)
        vb = sorted(list({b.replace("'", "") for b in vb}))
        vb = pd.DataFrame([b.split(',') for b in vb], 
                         columns=['agent', 'target', 'action'])
        vb['agent'] = vb['agent'].astype(str)
        vb['target'] = vb['target'].astype(str)
        vb['action'] = vb['action'].astype(str).str.lower()

        if traintest == 'train':
            try:
                annot = pd.read_parquet(
                    path.replace('train_tracking', 'train_annotation')
                )
            except FileNotFoundError:
                continue

        def _mk_meta(index, agent_id, target_id):
            m = pd.DataFrame({
                'lab_id': lab_id,
                'video_id': video_id,
                'agent_id': agent_id,
                'target_id': target_id,
                'video_frame': index.astype('int32', copy=False),
                'frames_per_second': np.float32(fps),
                'sleeping': sleeping,
                'arena_shape': arena_shape,
                'arena_width_cm': np.float32(arena_w),
                'arena_height_cm': np.float32(arena_h),
                'n_mice': np.int8(n_mice),
            })
            for c in ('lab_id', 'video_id', 'agent_id', 'target_id', 'arena_shape'):
                m[c] = m[c].astype('category')
            return m

        # Generate SINGLE mouse data
        if generate_single:
            vb_single = vb.query("target == 'self'")
            for agent_str in pd.unique(vb_single['agent']):
                col_lab = _resolve(agent_str)
                if col_lab is None:
                    continue
                actions = sorted(vb_single.loc[vb_single['agent'].eq(agent_str), 
                                               'action'].unique().tolist())
                if not actions:
                    continue

                single = pvid.loc[:, col_lab]
                meta_df = _mk_meta(single.index, agent_str, 'self')

                if traintest == 'train':
                    a_num = _to_num(col_lab)
                    y = pd.DataFrame(False, index=single.index.astype('int32', copy=False),
                                    columns=actions)
                    a_sub = annot.query("(agent_id == @a_num) & (target_id == @a_num)")
                    for i in range(len(a_sub)):
                        ar = a_sub.iloc[i]
                        a = str(ar.action).lower()
                        if a in y.columns:
                            y.loc[int(ar['start_frame']):int(ar['stop_frame']), a] = True
                    yield 'single', single, meta_df, y
                else:
                    yield 'single', single, meta_df, actions

        # Generate PAIR mouse data
        if generate_pair:
            vb_pair = vb.query("target != 'self'")
            if len(vb_pair) > 0:
                allowed_pairs = set(map(tuple, vb_pair[['agent', 'target']].itertuples(
                    index=False, name=None)))

                for agent_num, target_num in itertools.permutations(
                        np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    agent_str = f"mouse{_to_num(agent_num)}"
                    target_str = f"mouse{_to_num(target_num)}"
                    if (agent_str, target_str) not in allowed_pairs:
                        continue

                    a_col = _resolve(agent_str)
                    b_col = _resolve(target_str)
                    if a_col is None or b_col is None:
                        continue

                    actions = sorted(
                        vb_pair.query("(agent == @agent_str) & (target == @target_str)")
                        ['action'].unique().tolist()
                    )
                    if not actions:
                        continue

                    pair_xy = pd.concat([pvid[a_col], pvid[b_col]], 
                                       axis=1, keys=['A', 'B'])
                    meta_df = _mk_meta(pair_xy.index, agent_str, target_str)

                    if traintest == 'train':
                        a_num = _to_num(a_col)
                        b_num = _to_num(b_col)
                        y = pd.DataFrame(False, 
                                        index=pair_xy.index.astype('int32', copy=False),
                                        columns=actions)
                        a_sub = annot.query("(agent_id == @a_num) & (target_id == @b_num)")
                        for i in range(len(a_sub)):
                            ar = a_sub.iloc[i]
                            a = str(ar.action).lower()
                            if a in y.columns:
                                y.loc[int(ar['start_frame']):int(ar['stop_frame']), a] = True
                        yield 'pair', pair_xy, meta_df, y
                    else:
                        yield 'pair', pair_xy, meta_df, actions
