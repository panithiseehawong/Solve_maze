#!/usr/bin/env python3
"""Stage 2: select a saved maze, review cell walls, and save logical JSON."""

# ================= USER SETTINGS: edit here =================
MAP_FOLDER = '/home/yahboom/maze_robot/maps'  # File picker starts here.
MAZE_ROWS = 5  # Initial value in the size-selection window.
MAZE_COLS = 5
CELL_EAST_WEST_CM = 40.0       # Nominal spacing, recorded for later movement.  33.3 for 30 cm
CELL_NORTH_SOUTH_CM = 40.0      #  30 for 30 cm
MINIMUM_KNOWN_FRACTION = 0.50  # Required in every cell before acceptance.
EDGE_BAND_HALF_WIDTH_CELLS = 0.12
EDGE_ENDPOINT_TRIM_FRACTION = 0.20
WALL_FRACTION = 0.60
OPEN_FRACTION = 0.90
# Coordinates: (0,0) TOP-LEFT; x increases right; y increases down.
# No robot connection is required. This program sends no movement commands.
# ===========================================================

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

from exploration.map_region import load_map, rectify_map, cell_coverage, pixel_to_world
from maze_logic import DetectionSettings, STATES, detect_maze, edge_segment, reachable_cells


def choose_map_file():
    """Ask for a saved ROS map YAML; cancelling never chooses a fallback map."""
    import tkinter as tk
    from tkinter import filedialog

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            parent=root, title='Choose saved maze map: raw_map.yaml',
            initialdir=str(Path(MAP_FOLDER).expanduser()),
            filetypes=[('ROS map YAML', ('*.yaml', '*.yml')), ('All files', '*')])
        return selected or None
    except tk.TclError as error:
        raise RuntimeError('Cannot open the map file picker. Run in the VM desktop, or pass '
                           'the map YAML filename directly in the command.') from error
    finally:
        if root is not None:
            root.destroy()


def choose_maze_dimensions(rows, cols):
    """Choose conversion dimensions before selecting corners or detecting walls."""
    import tkinter as tk
    from tkinter import ttk

    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError('Cannot open the maze size window. Run in the VM desktop, or pass '
                           'both --rows and --cols in the command.') from error
    root.title('Convert map — choose size')
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=20)
    frame.grid()
    ttk.Label(frame, text='Set the grid size of your real maze.', font=('', 13)).grid(
        row=0, column=0, columnspan=2, pady=(0, 15))
    values = {}
    for row, (key, label, default) in enumerate((
            ('rows', 'Rows (top to bottom)', rows),
            ('cols', 'Columns (left to right)', cols)), 1):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky='w', padx=(0, 15), pady=5)
        variable = tk.StringVar(value=str(default))
        entry = ttk.Entry(frame, textvariable=variable, width=20)
        entry.grid(row=row, column=1, pady=5)
        if key == 'rows':
            entry.focus_set()
            entry.selection_range(0, 'end')
        values[key] = variable
    message = tk.StringVar(value='Next, select the four outer corners of the maze map.')
    ttk.Label(frame, textvariable=message, wraplength=420, foreground='#333333').grid(
        row=3, column=0, columnspan=2, pady=12)
    result = []

    def confirm():
        try:
            chosen = (int(values['rows'].get()), int(values['cols'].get()))
            if min(chosen) < 1:
                raise ValueError
        except ValueError:
            message.set('Rows and columns must be positive whole numbers (1 or greater).')
            return
        result.append(chosen)
        root.destroy()

    ttk.Button(frame, text='Continue to corner selection', command=confirm).grid(
        row=4, column=0, sticky='ew', padx=(0, 10), pady=5)
    ttk.Button(frame, text='Cancel', command=root.destroy).grid(row=4, column=1, sticky='ew', pady=5)
    root.bind('<Return>', lambda event: confirm())
    root.bind('<Escape>', lambda event: root.destroy())
    root.mainloop()
    return result[0] if result else None


def draw_maze(axis, maze, crop=None):
    """The image is stretched only for review; raw-map coordinates are retained."""
    axis.clear()
    if crop is not None:
        axis.imshow(crop, cmap='gray', vmin=0, vmax=255, interpolation='nearest',
                    extent=(0, maze.cols, maze.rows, 0))
    axis.set_xlim(-0.12, maze.cols + 0.12)
    axis.set_ylim(maze.rows + 0.12, -0.12)
    axis.set_aspect('equal')
    axis.set_xticks(np.arange(maze.cols) + 0.5)
    axis.set_xticklabels(range(maze.cols))
    axis.set_yticks(np.arange(maze.rows) + 0.5)
    axis.set_yticklabels(range(maze.rows))
    axis.set_xlabel('x → East/right')
    axis.set_ylabel('y → South/down')
    axis.plot([0, maze.cols, maze.cols, 0, 0], [0, 0, maze.rows, maze.rows, 0],
              color='black', linewidth=3)
    colors = {'WALL': 'tab:red', 'OPEN': 'tab:green', 'UNKNOWN': 'darkorange'}
    for x, y, side in maze.interior_edges():
        first, last = edge_segment(x, y, side)
        state = maze.cells[y][x][side]
        axis.plot([first[0], last[0]], [first[1], last[1]], color=colors[state],
                  linewidth=3 if state == 'WALL' else 2,
                  linestyle='-' if state == 'WALL' else '--')
    if maze.rows * maze.cols <= 100:
        for y in range(maze.rows):
            for x in range(maze.cols):
                axis.text(x + 0.5, y + 0.5, f'({x},{y})', ha='center', va='center',
                          fontsize=max(6, min(12, 42 / max(maze.rows, maze.cols))),
                          bbox={'facecolor': 'white', 'alpha': 0.65, 'edgecolor': 'none'})


def nearest_edge(maze, x, y, tolerance=0.18):
    """Find an interior boundary clicked in cell coordinates; outer edges lock."""
    if x is None or y is None or not (0 <= x <= maze.cols and 0 <= y <= maze.rows):
        return None
    best, distance = None, tolerance
    for cx, cy, side in maze.interior_edges():
        first, last = edge_segment(cx, cy, side)
        if side == 'E':
            gap = math.hypot(x - first[0], y - min(max(y, first[1]), last[1]))
        else:
            gap = math.hypot(y - first[1], x - min(max(x, first[0]), last[0]))
        if gap < distance:
            best, distance = (cx, cy, side), gap
    return best


def acceptance_problems(maze, crop, minimum_known):
    maze.validate()
    problems = []
    unknown = maze.unknown_edges()
    if unknown:
        problems.append(f'{len(unknown)} UNKNOWN interior edge(s) remain')
    coverage = cell_coverage(crop, maze.rows, maze.cols)
    incomplete = [(x, y) for y, row in enumerate(coverage) for x, value in enumerate(row)
                  if value < minimum_known]
    if incomplete:
        problems.append(f'{len(incomplete)} cell(s) have less than {minimum_known:.0%} mapped area')
    reachable = len(reachable_cells(maze))
    if reachable != maze.rows * maze.cols:
        problems.append(f'only {reachable}/{maze.rows * maze.cols} cells connect through OPEN edges')
    return problems


def save_conversion(map_yaml, metadata, original, crop, corners, transform, maze,
                    evidence, settings, minimum_known, spacing_cm, accepted,
                    selection_method='human_four_corner_selection'):
    maze.validate(complete=accepted)
    problems = acceptance_problems(maze, crop, minimum_known)
    if accepted and problems:
        raise ValueError('Cannot accept: ' + '; '.join(problems))
    if not all(math.isfinite(value) and value > 0 for value in spacing_cm):
        raise ValueError('Cell spacing must be positive and finite.')
    folder = Path(map_yaml).resolve().parent / (
        'logical_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(exist_ok=False)
    record = {
        'schema': 'thin_wall_maze_v1', 'rows': maze.rows, 'cols': maze.cols,
        'coordinate_convention': {'origin': 'top_left', 'x_increases': 'East', 'y_increases': 'South'},
        'cell_storage': 'cells[y][x], each with N/E/S/W states',
        'cells': maze.cells,
        'cell_spacing_m': {'east_west': spacing_cm[0] / 100, 'north_south': spacing_cm[1] / 100},
        'accepted_complete': bool(accepted),
        'human_confirmed_cells_and_walls': bool(accepted),
        'unknown_interior_edges': [list(edge) for edge in maze.unknown_edges()],
        'acceptance_problems': problems,
        'outer_boundary_policy': 'WALL: movement outside the selected grid is forbidden, even at physical gaps.',
        'source': {'raw_map_yaml': str(Path(map_yaml).resolve()), 'raw_map_metadata': metadata,
                   'corners_image_pixels': corners, 'selection_method': selection_method,
                   'corner_order': ['top_left', 'top_right', 'bottom_right', 'bottom_left'],
                   'corners_world_m': [pixel_to_world(point, original.shape[0], metadata) for point in corners],
                   'crop_pixel_to_raw_pixel_perspective': transform, 'crop_image': 'maze_crop.png'},
        'detection_settings': asdict(settings),
        'minimum_known_fraction': minimum_known,
        'known_fraction_per_cell': cell_coverage(crop, maze.rows, maze.cols),
        'edge_detection': [dict(item, final_state=maze.cells[item['cell'][1]][item['cell'][0]][item['side']],
                                manually_changed=(maze.cells[item['cell'][1]][item['cell'][0]][item['side']]
                                                  != item['detected_state'])) for item in evidence],
        'notes': 'A candidate is not ready for robot solving. Only human-confirmed complete files should be used. '
                 'Spacing is nominal, not a measurement derived from SLAM. The raw map remains unchanged.',
    }
    Image.fromarray(crop).save(folder / 'maze_crop.png')
    (folder / 'logical_maze.json').write_text(json.dumps(record, indent=2) + '\n')
    return folder


def preview_figure(maze, crop):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 7))
    figure.subplots_adjust(bottom=0.22, top=0.80)
    draw_maze(axes[0], maze, crop)
    draw_maze(axes[1], maze)
    axes[0].set_title('Saved map + detected cell edges')
    axes[1].set_title('Logical maze: red WALL / green OPEN / orange UNKNOWN')
    return figure, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map_yaml', nargs='?',
                        help='Saved ROS map YAML; omit to choose it in a file picker.')
    parser.add_argument('--rows', type=int, help='Maze rows; specify both dimensions to skip the size window.')
    parser.add_argument('--cols', type=int, help='Maze columns; otherwise choose dimensions in the size window.')
    args = parser.parse_args()
    settings = DetectionSettings(EDGE_BAND_HALF_WIDTH_CELLS, EDGE_ENDPOINT_TRIM_FRACTION,
                                 WALL_FRACTION, OPEN_FRACTION)
    try:
        settings.validate()
        if any(value is not None and value < 1 for value in (args.rows, args.cols)) or not 0 < MINIMUM_KNOWN_FRACTION <= 1:
            raise ValueError('Positive dimensions and minimum-known in (0,1] are required.')
        if not all(math.isfinite(value) and value > 0 for value in
                   (CELL_EAST_WEST_CM, CELL_NORTH_SOUTH_CM)):
            raise ValueError('Cell spacing must be positive and finite.')
        if args.map_yaml is None:
            print('Choose a saved map YAML in the file picker. Its image loads automatically.', flush=True)
            args.map_yaml = choose_map_file()
            if args.map_yaml is None:
                print('Cancelled. No map selected; nothing saved.')
                return
        metadata, original = load_map(args.map_yaml)
        if args.rows is None or args.cols is None:
            dimensions = choose_maze_dimensions(args.rows if args.rows is not None else MAZE_ROWS,
                                                args.cols if args.cols is not None else MAZE_COLS)
            if dimensions is None:
                print('Cancelled. No logical maze saved.')
                return
            args.rows, args.cols = dimensions
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.error(str(error))

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    print(f'Stage 2: {args.cols} columns × {args.rows} rows. (0,0) is TOP-LEFT; y increases DOWN.\n'
          f'Source: {Path(args.map_yaml).resolve()}\n'
          'Select the four OUTER wall-line corners, not the internal divider.\n'
          'No robot connection is needed; no movement commands are sent.', flush=True)
    while True:
        figure, axis = plt.subplots(figsize=(9, 8))
        axis.imshow(original, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        yy, xx = np.where(original == 0)
        if len(xx):
            pad = max(3, int(max(np.ptp(xx), np.ptp(yy)) * 0.1))
            axis.set_xlim(max(0, xx.min() - pad), min(original.shape[1] - 1, xx.max() + pad))
            axis.set_ylim(min(original.shape[0] - 1, yy.max() + pad), max(0, yy.min() - pad))
        axis.set_title('Click wall-line corners: TOP-LEFT → TOP-RIGHT → BOTTOM-RIGHT → BOTTOM-LEFT\n'
                       'Choose where the outer wall lines meet; the grid divides this region into equal cells.\n'
                       'Close this window to cancel.')
        corners = [[float(x), float(y)] for x, y in plt.ginput(4, timeout=0, show_clicks=True)]
        plt.close(figure)
        if len(corners) != 4:
            print('Cancelled. No logical maze saved.')
            return
        try:
            crop, transform = rectify_map(original, corners, args.rows, args.cols)
            maze, evidence = detect_maze(crop, args.rows, args.cols, settings)
        except ValueError as error:
            print(f'{error} Select the corners again.', flush=True)
            continue

        figure, axes = preview_figure(maze, crop)
        decision = {'action': 'cancel'}
        button_axes = [figure.add_axes(rect) for rect in
                       ([0.07, 0.04, 0.25, 0.07], [0.37, 0.04, 0.25, 0.07], [0.67, 0.04, 0.25, 0.07])]
        buttons = [Button(button_axes[0], 'Confirm & save'), Button(button_axes[1], 'Save candidate'),
                   Button(button_axes[2], 'Reselect corners')]

        def refresh():
            for axis, image in zip(axes, (crop, None)):
                draw_maze(axis, maze, image)
            axes[0].set_title('Saved map + cell edges; click an interior edge to correct it')
            axes[1].set_title('Red WALL / green OPEN / orange UNKNOWN')
            problems = acceptance_problems(maze, crop, MINIMUM_KNOWN_FRACTION)
            figure.suptitle('Click an interior edge: WALL → OPEN → UNKNOWN → WALL. Outer edges stay closed.\n'
                           'Confirm only after checking EVERY cell and wall against the real maze.\n'
                           + ('Cannot accept: ' + '; '.join(problems) if problems
                              else 'Checks passed. Confirm & save records YOUR confirmation that the maze is correct.'),
                           fontsize=10)
            button_axes[0].set_facecolor('lightgray' if problems else 'palegreen')
            figure.canvas.draw_idle()

        def click(event):
            if event.inaxes not in axes or event.button != 1:
                return
            if figure.canvas.toolbar is not None and figure.canvas.toolbar.mode:
                return
            edge = nearest_edge(maze, event.xdata, event.ydata)
            if edge is None:
                return
            x, y, side = edge
            state = maze.cells[y][x][side]
            maze.set_edge(x, y, side, STATES[(STATES.index(state) + 1) % len(STATES)])
            print(f'Cell ({x},{y}) {side}: {state} → {maze.cells[y][x][side]} (neighbor updated too)', flush=True)
            refresh()

        def finish(action):
            if action == 'accept':
                problems = acceptance_problems(maze, crop, MINIMUM_KNOWN_FRACTION)
                if problems:
                    print('Cannot accept: ' + '; '.join(problems), flush=True)
                    return
            decision['action'] = action
            plt.close(figure)

        figure.canvas.mpl_connect('button_press_event', click)
        buttons[0].on_clicked(lambda event: finish('accept'))
        buttons[1].on_clicked(lambda event: finish('candidate'))
        buttons[2].on_clicked(lambda event: finish('reselect'))
        refresh()
        plt.show()
        if decision['action'] == 'reselect':
            continue
        if decision['action'] == 'cancel':
            print('Cancelled. No logical maze saved.')
            return
        folder = save_conversion(args.map_yaml, metadata, original, crop, corners, transform,
                                 maze, evidence, settings, MINIMUM_KNOWN_FRACTION,
                                 (CELL_EAST_WEST_CM, CELL_NORTH_SOUTH_CM), decision['action'] == 'accept')
        figure.savefig(folder / 'logical_preview.png', dpi=140)
        print(('Confirmed logical maze: ' if decision['action'] == 'accept' else 'Unaccepted candidate: ')
              + str(folder / 'logical_maze.json'), flush=True)
        print('Stage 2 finished. No robot movement was performed.', flush=True)
        return


if __name__ == '__main__':
    main()
