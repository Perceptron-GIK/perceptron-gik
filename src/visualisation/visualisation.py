import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from src.Constants.char_to_key import (
    ALL_CHARS,
    ALL_CHARS_4,
    ALL_CHARS_DIAGONAL,
    CHAR_TO_INDEX,
    CHAR_TO_INDEX_4,
    CHAR_TO_INDEX_DIAGONAL,
    FULL_COORDS,
    FULL_COORDS_4,
    FULL_COORDS_DIAGONAL,
    INDEX_TO_CHAR,
    INDEX_TO_CHAR_4,
    INDEX_TO_CHAR_DIAGONAL,
    KEY_COORDS,
    NUM_CLASSES,
    NUM_CLASSES_4,
    NUM_CLASSES_DIAGONAL,
    SPACE_ANCHORS,
    SPECIAL_COORDS,
)


DISPLAY_TOKENS = {"\n": "ENTER", " ": "SPACE", "\t": "TAB", "\b": "BKSP"}


def _unpack_sample(sample):
    """Return (x, y) from dataset samples that may include class_id."""
    if len(sample) == 3:
        x, y, _ = sample
    else:
        x, y = sample
    return x, y


def _to_class_index(label) -> int:
    """Convert either one-hot/vector or scalar tensor labels to class index."""
    if isinstance(label, torch.Tensor) and label.ndim > 0:
        return int(label.argmax().item())
    if isinstance(label, torch.Tensor):
        return int(label.item())
    return int(label)


def _print_prediction_results(results, subset_name, max_samples):
    """Print consistent prediction summary for both modes."""
    n_total = len(results)
    correct = sum(1 for _, _, ok in results if ok)
    acc = (correct / n_total) if n_total else 0.0
    print(f"=== {subset_name} Predictions ({correct}/{n_total} correct = {acc:.2%}) ===")
    print()

    n_show = min(max_samples, n_total)
    for true_char, pred_char, is_correct in results[:n_show]:
        true_disp = DISPLAY_TOKENS.get(true_char, true_char)
        pred_disp = DISPLAY_TOKENS.get(pred_char, pred_char)
        match = "OK" if is_correct else "X"
        print(f"{match} True: {true_disp:8} | Pred: {pred_disp:8}")
    if n_total > n_show:
        print(f"... and {n_total - n_show} more samples")
    print()


def get_closest_coordinate(coord, coord_dict=FULL_COORDS):
    """Return the character with minimum squared distance to coord."""
    if isinstance(coord, torch.Tensor):
        coord = coord.detach().squeeze().cpu().numpy()
    coord = np.asarray(coord)
    best_char, best_dist = None, float("inf")
    for char, (cx, cy) in coord_dict.items():
        d = (coord[0] - cx) ** 2 + (coord[1] - cy) ** 2
        if d < best_dist:
            best_dist, best_char = d, char
    return best_char if best_char is not None else "?"


def postprocess_coordinate_output(
    y_pred: torch.Tensor,
    coord_scale=(9.0, 4.0),
    apply_sigmoid: bool = True,
) -> torch.Tensor:
    """Map raw coordinate logits to keyboard-coordinate space."""
    if apply_sigmoid:
        y_pred = torch.sigmoid(y_pred)
    scale = y_pred.new_tensor(coord_scale)
    return y_pred * scale


def show_predictions_coordinate(
    data_subset,
    model,
    device,
    subset_name,
    coord_dict=FULL_COORDS,
    max_samples=50,
    coord_scale=(10.0, 4.0),
    apply_sigmoid=True,
):
    """Show predictions for coordinate mode by nearest key lookup."""
    model.eval()
    results = []
    dataset = data_subset.dataset
    indices = data_subset.indices

    with torch.no_grad():
        for i in range(len(data_subset)):
            x, _ = _unpack_sample(data_subset[i])
            true_char = dataset._labels[indices[i]]
            x = x.unsqueeze(0).to(device)
            y_pred = model(x)
            y_pred = postprocess_coordinate_output(
                y_pred,
                coord_scale=coord_scale,
                apply_sigmoid=apply_sigmoid,
            )
            pred_char = get_closest_coordinate(y_pred, coord_dict)

            is_correct = true_char == pred_char
            results.append((true_char, pred_char, is_correct))

    _print_prediction_results(results, subset_name, max_samples)


def show_predictions(
    data_subset,
    model,
    device,
    subset_name,
    max_samples=50,
    index_to_char=None,
):
    """Show predictions for classification mode. For 4-class, pass index_to_char=INDEX_TO_CHAR_4."""
    index_to_char = index_to_char or INDEX_TO_CHAR
    model.eval()
    results = []

    with torch.no_grad():
        for idx in range(len(data_subset)):
            x, y_true = _unpack_sample(data_subset[idx])
            x = x.unsqueeze(0).to(device)
            y_pred = model(x)

            true_idx = _to_class_index(y_true)
            pred_idx = y_pred.argmax().item()

            true_char = index_to_char.get(true_idx, "?")
            pred_char = index_to_char.get(pred_idx, "?")

            is_correct = true_char == pred_char
            results.append((true_char, pred_char, is_correct))

    _print_prediction_results(results, subset_name, max_samples)


def key_distance(ch1, ch2, coord_dict=None):
    """
    Distance between two keys.
    For space (' '), use the minimum distance to its multiple anchor points
    to mimic a long bar; other keys use their single coordinate.
    """
    coord_dict = coord_dict or FULL_COORDS

    def coords(ch):
        if ch == ' ':
            return SPACE_ANCHORS
        elif ch in coord_dict:
            return [coord_dict[ch]]
        return [(0.0, 0.0)]

    pts1 = coords(ch1)
    pts2 = coords(ch2)

    d_min = float('inf')
    for (x1, y1) in pts1:
        for (x2, y2) in pts2:
            d = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
            if d < d_min:
                d_min = d
    return d_min


def make_display_label(c):
    if c == ' ':
        return 'SP'
    elif c == '\n':
        return 'ENT'
    elif c == '\b':
        return 'BS'
    elif c == '\t':
        return 'TAB'
    else:
        return c


def compute_confusion_matrix_40x40(
    data_subset,
    model,
    device,
    coord_dict=None,
    coord_scale=(9.0, 4.0),
    apply_sigmoid=True,
    num_classes=None,
    char_to_index=None,
):
    """Return NxN confusion matrix. If coord_dict is None, labels are class indices/one-hot; else coordinate mode.
    For 4-class mode, pass num_classes=4 and char_to_index=CHAR_TO_INDEX_4."""
    num_classes = num_classes or NUM_CLASSES
    char_to_index = char_to_index or CHAR_TO_INDEX
    model.eval()
    y_true_list = []
    y_pred_list = []
    dataset = data_subset.dataset
    indices = data_subset.indices

    with torch.no_grad():
        for i in range(len(data_subset)):
            x, y = _unpack_sample(data_subset[i])
            x = x.unsqueeze(0).to(device)
            y_pred = model(x)

            if coord_dict is not None:
                y_pred = postprocess_coordinate_output(
                    y_pred,
                    coord_scale=coord_scale,
                    apply_sigmoid=apply_sigmoid,
                )
                true_char = dataset._labels[indices[i]]
                pred_char = get_closest_coordinate(y_pred, coord_dict)
                y_true_list.append(char_to_index.get(true_char, 0))
                y_pred_list.append(char_to_index.get(pred_char, 0))
            else:
                y_true_list.append(_to_class_index(y))
                y_pred_list.append(y_pred.argmax().item())

    labels_idx = list(range(num_classes))
    cm_orig = confusion_matrix(y_true_list, y_pred_list, labels=labels_idx)
    return cm_orig

def plot_confusion_matrix_40x40(cm_orig, subset_name, num_classes=None, index_to_char=None):
    """Plot the confusion matrix. For 4-class, pass num_classes=4 and index_to_char=INDEX_TO_CHAR_4."""
    num_classes = num_classes or NUM_CLASSES
    index_to_char = index_to_char or INDEX_TO_CHAR
    n = min(num_classes, cm_orig.shape[0])
    display_labels = [make_display_label(index_to_char[i]) for i in range(n)]

    figsize = (16, 14) if n > 4 else (8, 6)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_orig[:n, :n],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
        annot_kws={'size': 10 if n <= 4 else 7}
    )

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(
        f'{subset_name} Confusion Matrix ({n} classes, raw counts)',
        fontsize=14,
        pad=40
    )

    plt.xticks(rotation=45, ha='right', fontsize=10 if n <= 4 else 8)
    plt.yticks(rotation=0, fontsize=10 if n <= 4 else 8)
    plt.tight_layout()
    plt.show()


def plot_anchor_with_closest_neighbours(
    cm_orig, anchor_char, subset_name, k_neighbours=6,
    num_classes=None, index_to_char=None, full_coords=None, all_chars=None,
):
    """
    For a given anchor_char, find its k_neighbours closest keys (by keyboard distance),
    and plot a confusion submatrix. For 4-class, pass num_classes=4, index_to_char=INDEX_TO_CHAR_4,
    full_coords=FULL_COORDS_4, all_chars=ALL_CHARS_4.
    """
    num_classes = num_classes or NUM_CLASSES
    index_to_char = index_to_char or INDEX_TO_CHAR
    full_coords = full_coords or FULL_COORDS
    all_chars = all_chars or ALL_CHARS

    if anchor_char not in full_coords:
        raise ValueError(f'No coords for anchor_char {anchor_char!r}')

    char_to_idx = {ch: idx for idx, ch in index_to_char.items()}
    others = [ch for ch in all_chars if ch != anchor_char]
    def _dist(ch):
        return key_distance(anchor_char, ch, full_coords)
    others_sorted = sorted(others, key=_dist)
    k_neighbours = min(k_neighbours, len(others_sorted))
    closest_chars = others_sorted[:k_neighbours]

    ordered_chars = [anchor_char] + closest_chars
    ordered_indices = [char_to_idx[ch] for ch in ordered_chars]
    ordered_display = [make_display_label(ch) for ch in ordered_chars]

    cm_sub = cm_orig[np.ix_(ordered_indices, ordered_indices)]

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        cm_sub,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=ordered_display,
        yticklabels=ordered_display,
        ax=ax,
        annot_kws={'size': 10}
    )

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(
        f'{subset_name}: "{anchor_char}" + {k_neighbours} closest keys',
        fontsize=14,
        pad=40
    )

    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.show()

    return closest_chars, cm_sub


def plot_virtual_keyboard_heatmap(cm_orig, anchor_char, subset_name, num_classes=None, index_to_char=None, char_to_index=None):
    """For 4-class, pass num_classes=4, index_to_char=INDEX_TO_CHAR_4, char_to_index=CHAR_TO_INDEX_4.
    anchor_char can be class label (e.g. 'asdfghjkl') or single char (e.g. 's') mapped to its class."""
    num_classes = num_classes or NUM_CLASSES
    index_to_char = index_to_char or INDEX_TO_CHAR
    char_to_index = char_to_index or CHAR_TO_INDEX
    char_to_idx = {ch: idx for idx, ch in index_to_char.items()}
    char_to_class_idx = {c: idx for idx, s in index_to_char.items() for c in s}
    i_anchor = char_to_idx.get(anchor_char) or char_to_class_idx.get(anchor_char)
    if i_anchor is None:
        raise ValueError(f'Unknown anchor_char {anchor_char!r}')

    n = min(num_classes, cm_orig.shape[0])
    row = cm_orig[i_anchor].astype(float)[:n]
    # Map each physical key to its class value for heatmap coloring
    key_values = {ch: row[char_to_index.get(ch, 0)] if char_to_index.get(ch, 0) < n else 0.0
                  for ch in list(KEY_COORDS.keys()) + list(SPECIAL_COORDS.keys()) + [' ']}

    all_vals = np.array(list(key_values.values()))
    vmax = all_vals.max() if np.any(all_vals > 0) else 1.0
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.Reds

    fig, ax = plt.subplots(figsize=(10, 5))

    key_width = 0.8
    key_height = 0.8

    def draw_key(x, y, label, value, zorder=2, fontsize=10):
        color = cmap(norm(value)) if value > 0 else (0.9, 0.9, 0.9, 1.0)
        rect = plt.Rectangle(
            (x - key_width/2, y - key_height/2),
            key_width,
            key_height,
            edgecolor='k',
            facecolor=color,
            zorder=zorder
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, zorder=3)

    for ch, (x, y) in KEY_COORDS.items():
        draw_key(x, y, ch.upper(), key_values.get(ch, 0.0), zorder=2, fontsize=10)

    for ch, (x, y) in SPECIAL_COORDS.items():
        if ch == ' ':
            continue
        label = {'\n': 'ENT', '\b': 'BS', '\t': 'TAB'}.get(ch, ch)
        draw_key(x, y, label, key_values.get(ch, 0.0), zorder=2, fontsize=9)

    xs = [p[0] for p in SPACE_ANCHORS]
    y_space = SPACE_ANCHORS[0][1]
    x_min, x_max = min(xs) - 0.4, max(xs) + 0.4
    space_width = x_max - x_min
    space_height = 0.8

    v_space = key_values.get(' ', 0.0)
    color_space = cmap(norm(v_space)) if v_space > 0 else (0.9, 0.9, 0.9, 1.0)

    space_rect = plt.Rectangle(
        (x_min, y_space - space_height/2),
        space_width,
        space_height,
        edgecolor='k',
        facecolor=color_space,
        zorder=1
    )
    ax.add_patch(space_rect)
    ax.text((x_min + x_max) / 2, y_space, 'SP', ha='center', va='center', fontsize=10, zorder=3)

    # Highlight anchor with thick border.
    def outline_rect(x, y, w, h, lw=2.5, color='blue'):
        outline = plt.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            edgecolor=color,
            linewidth=lw,
            zorder=4
        )
        ax.add_patch(outline)

    def outline_anchor(ch, lw=2.5, color='blue'):
        if ch == ' ':
            outline_rect(x_min, y_space - space_height/2, space_width, space_height, lw=lw, color=color)
        elif ch in KEY_COORDS:
            x, y = KEY_COORDS[ch]
            outline_rect(x - key_width/2, y - key_height/2, key_width, key_height, lw=lw, color=color)
        elif ch in SPECIAL_COORDS:
            x, y = SPECIAL_COORDS[ch]
            outline_rect(x - key_width/2, y - key_height/2, key_width, key_height, lw=lw, color=color)

    # Outline anchor: for single-char anchor, outline that key; for class label, outline first char
    outline_ch = anchor_char if anchor_char in KEY_COORDS or anchor_char in SPECIAL_COORDS or anchor_char == ' ' else (anchor_char[0] if anchor_char and anchor_char[0] in KEY_COORDS else anchor_char)
    outline_anchor(outline_ch)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Predicted count')

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.0, 13.0)
    ax.set_ylim(-1, 5.0)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'GIK Keyboard Heatmap Prediction when True Key = "{anchor_char}", {subset_name} set',
                 fontsize=14)
    plt.tight_layout()
    plt.show()
