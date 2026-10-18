#!/usr/bin/env python3
"""Stage 3: A* junction solving with human confirmations and manual recovery."""

# ================= USER SETTINGS: edit these numbers =================
# Main physical solver: python3 ~/maze_robot/solve_maze.py --robot
# Distances use CENTIMETERS. Restart the solver after editing.
LOGICAL_MAZE_JSON = None       # None opens a file picker; otherwise use this file.
LOGICAL_MAZE_FOLDER = '/home/yahboom/maze_robot/maps'  # Picker starts here.
DEFAULT_START = (0, 0)
DEFAULT_FINISH = None          # None means bottom-right: (columns-1, rows-1).
INITIAL_HEADING = 'East'       # GUI default: North, East, South, or West.
CELL_EAST_WEST_CM = None       # Robot mode: None uses spacing from the chosen logical maze.
CELL_NORTH_SOUTH_CM = None

FORWARD_SPEED_CM_S = 15.0    #10.0   # Maximum forward speed, centimeters per second.
TURN_SPEED_RAD_S = 0.60   #0.60   # Maximum in-place rotation speed, radians per second.
TURN_TOLERANCE_DEG = 3.0      # Accepted heading error at the end of a turn.
FIXED_JUNCTION_TURNS = True  # Left/right = 90 degrees from current yaw; no LiDAR angle adjustment or partial entry.
TURN_GAIN = 2.0              # Friend's proportional rotation control.
TURN_MIN_SPEED_RAD_S = 0.20  # Minimum turning speed to overcome motor deadband (capped by TURN_SPEED_RAD_S).
TURN_EXIT_SPEED_CM_S = 5.0   # Slow exit after alignment; left/right LiDAR steers toward the middle.
TURN_EXIT_S = 1.0            # Active seconds at the slow exit speed; junction checks stay active.
ENTRY_SPEED_CM_S = 4.0      # Forward speed at a partial turn angle, before corridor alignment.
ENTRY_LOOKAHEAD_CM = 25.0   # Planning/finish-turn lookahead only; not a gate on each entry movement.
ENTRY_STEP_CM = 3.0        # Short forward clearance check; enlarged if speed/sensor timeout requires it.
ENTRY_TIMEOUT_S = 20.0     # Active-time limit for finding room to finish the corridor turn.

RECENTER_TURN_DEG = 5.0       # Automatic small turn away from a close side wall; tries half if needed.
RECENTER_SPEED_CM_S = 4.0     # Slow forward motion after the corrective turn.
RECENTER_TURN_RAD_S = 0.20    # Corrective rotation speed, capped by TURN_SPEED_RAD_S.
RECENTER_TIMEOUT_S = 8.0     # Stop if side clearance does not recover.
RECENTER_MIN_GAP_CM = 0.5    # Minimum measured/predicted body-edge gap during correction.
RECENTER_RELEASE_EXTRA_CM = 0.5 # Recover normal body margin plus this before normal driving.

LIDAR_ALIGN_TOLERANCE_DEG = 1.5  # Estimated-angle mode only (FIXED_JUNCTION_TURNS = False).
LIDAR_ALIGN_TIMEOUT_S = 5.0      # Also bounds fresh forward-passage verification after a fixed turn.
LIDAR_ALIGN_SPEED_RAD_S = 0.30   # Maximum speed for small in-place corrections.
LIDAR_ALIGN_MIN_SPEED_RAD_S = 0.20 # Minimum correction speed to overcome motor deadband.
LIDAR_ALIGN_GAIN = 2.0
LIDAR_ALIGN_SETTLE_S = 0.30      # Stay aligned across fresh scans while stopped.
LIDAR_ALIGN_MAX_CORRECTION_DEG = 10.0 # Larger errors require manual recovery.

JUNCTION_SCAN_WAIT_S = 12.0  # Stop for recovery if a landmark cannot be verified.
ODOMETRY_STEP_SLACK_CM = 4.0 # Extra position change before a warning; this check does not stop motion.
MANUAL_SPEED_CM_S = 5.0     # Recovery only: w/s forward/backward.
MANUAL_TURN_RAD_S = 0.35    # Recovery only: a/d rotation in place.
MANUAL_KEY_TIMEOUT_S = 0.25 # Movement expires without a repeated/tapped key.

BODY_LENGTH_CM = 17.0         # Include wheels and protrusions; base at body center.
BODY_WIDTH_CM = 15.0
DRIVE_BODY_MARGIN_CM = 1.5    # Normal BODY EDGE gap; a close side wall can trigger automatic centering.
CLEARANCE_LIMIT_CM = 10.0     # ROTATION ONLY: nearest wall-point distance from base frame.
FRONT_STOP_CM = 13.5          # Forward-sector stop distance from base frame.
# Rotation/front limits are NOT gaps from the body edge.

WALL_THRESHOLD_CM = 30.0      # At/below this distance, a side is WALL.   19 for 30 cm
OPEN_THRESHOLD_CM = 40.0      # At/above this distance, a side is OPEN.   23 for 30 cm
# Between WALL and OPEN thresholds, or missing readings: UNKNOWN (wait stopped).

CENTER_TOLERANCE_CM = 1.5     # Allowed remaining distance to estimated cell center.
LATERAL_LIMIT_CM = 3.0        # Older --cell-mode only; junction mode ignores odometry sideways drift.
# Solver tuning is independent of USER SETTINGS in explore.py.
# Maze dimensions come from the selected map; start/finish are chosen in the GUI.
# Normal command previews the main junction method; --robot drives it.
# --cell-mode --robot explicitly requests the older cell-distance controller.
# ====================================================================

import argparse
import math
from pathlib import Path
import textwrap

from PIL import Image
import numpy as np

from convert_map import draw_maze
from maze_planner import (CONFIRMATION_KEYS, HEADING_NAMES, EndpointSelection, RecoverySelection,
                          heading_side, junction_plan, load_logical_maze)


def solver_controller_settings():
    """Convert the solver's editable centimeter settings to controller units."""
    return {
        'forward_speed_mps': FORWARD_SPEED_CM_S / 100,
        'turn_speed_rps': TURN_SPEED_RAD_S,
        'yaw_tolerance_deg': TURN_TOLERANCE_DEG,
        'body_length_m': BODY_LENGTH_CM / 100,
        'body_width_m': BODY_WIDTH_CM / 100,
        'drive_body_margin_m': DRIVE_BODY_MARGIN_CM / 100,
        'clearance_stop_m': CLEARANCE_LIMIT_CM / 100,
        'front_stop_m': FRONT_STOP_CM / 100,
        'wall_distance_m': WALL_THRESHOLD_CM / 100,
        'open_distance_m': OPEN_THRESHOLD_CM / 100,
        'center_tolerance_m': CENTER_TOLERANCE_CM / 100,
        'lateral_limit_m': LATERAL_LIMIT_CM / 100,
    }


def solver_junction_settings():
    from junction_solver import JunctionSettings
    return JunctionSettings(fixed_turns=FIXED_JUNCTION_TURNS,
                            observation_wait_s=JUNCTION_SCAN_WAIT_S,
                            turn_gain=TURN_GAIN,
                            turn_min_speed_rps=TURN_MIN_SPEED_RAD_S,
                            turn_exit_speed_mps=TURN_EXIT_SPEED_CM_S / 100,
                            turn_exit_s=TURN_EXIT_S,
                            entry_speed_mps=ENTRY_SPEED_CM_S / 100,
                            entry_lookahead_m=ENTRY_LOOKAHEAD_CM / 100,
                            entry_step_m=ENTRY_STEP_CM / 100,
                            entry_timeout_s=ENTRY_TIMEOUT_S,
                            recenter_turn_deg=RECENTER_TURN_DEG,
                            recenter_speed_mps=RECENTER_SPEED_CM_S / 100,
                            recenter_turn_rps=RECENTER_TURN_RAD_S,
                            recenter_timeout_s=RECENTER_TIMEOUT_S,
                            recenter_min_gap_m=RECENTER_MIN_GAP_CM / 100,
                            recenter_release_margin_m=RECENTER_RELEASE_EXTRA_CM / 100,
                            odometry_step_slack_m=ODOMETRY_STEP_SLACK_CM / 100,
                            manual_speed_mps=MANUAL_SPEED_CM_S / 100,
                            manual_turn_rps=MANUAL_TURN_RAD_S,
                            manual_key_timeout_s=MANUAL_KEY_TIMEOUT_S,
                            align_tolerance_deg=LIDAR_ALIGN_TOLERANCE_DEG,
                            align_timeout_s=LIDAR_ALIGN_TIMEOUT_S,
                            align_speed_rps=LIDAR_ALIGN_SPEED_RAD_S,
                            align_min_speed_rps=LIDAR_ALIGN_MIN_SPEED_RAD_S,
                            align_gain=LIDAR_ALIGN_GAIN,
                            align_settle_s=LIDAR_ALIGN_SETTLE_S,
                            align_max_correction_deg=LIDAR_ALIGN_MAX_CORRECTION_DEG)


def choose_logical_maze_file():
    """Choose a Stage 2 JSON file without automatically substituting another map."""
    import tkinter as tk
    from tkinter import filedialog

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            parent=root, title='Choose logical maze: confirmed map or simulation example',
            initialdir=str(Path(LOGICAL_MAZE_FOLDER).expanduser()),
            filetypes=[('Logical maze JSON', '*.json'), ('All files', '*')])
        return selected or None
    except tk.TclError as error:
        raise RuntimeError('Cannot open the logical-maze file picker. Run in the VM desktop, '
                           'or pass the logical_maze.json filename directly in the command.') from error
    finally:
        if root is not None:
            root.destroy()


def clicked_cell(maze, x, y):
    if x is None or y is None or not (0 <= x < maze.cols and 0 <= y < maze.rows):
        return None
    return math.floor(x), math.floor(y)


def preview_cell_spacing(record):
    """Use the same cell pitch overrides as robot mode; never invent spacing."""
    saved = record.get('cell_spacing_m', {})
    if not isinstance(saved, dict):
        raise ValueError('cell_spacing_m must contain east_west and north_south values.')
    east_west = CELL_EAST_WEST_CM / 100 if CELL_EAST_WEST_CM is not None else saved.get('east_west')
    north_south = CELL_NORTH_SOUTH_CM / 100 if CELL_NORTH_SOUTH_CM is not None else saved.get('north_south')
    if east_west is None and north_south is None:
        return None
    spacing = {'east_west': east_west, 'north_south': north_south}
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in spacing.values()):
        raise ValueError('Cell spacing must provide positive east_west and north_south values.')
    return spacing


def build_window(maze, map_path, crop=None, *, simulation_only=False, robot_mode=False,
                 junction_preview=None, recovery_from=None, on_recovery=None, cell_spacing_m=None):
    """Build the window without running its event loop, so it can be checked offline."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, Button

    if junction_preview is None:
        junction_preview = not robot_mode
    junction_robot = junction_preview and robot_mode
    recovering = recovery_from is not None
    if recovering:
        if robot_mode or not junction_preview:
            raise ValueError('Recovery selection requires the junction preview.')
        cell, direction, finish = recovery_from
        selection = RecoverySelection(maze, cell, finish, direction, cell_spacing_m=cell_spacing_m)
    else:
        selection = EndpointSelection(maze, DEFAULT_START, DEFAULT_FINISH, INITIAL_HEADING)
    position_label = 'Current cell' if recovering else 'Start'
    heading_label = 'Car facing' if recovering else 'Initial direction'
    figure = plt.figure(figsize=(12, 8.5))
    axis = figure.add_axes([0.07, 0.23, 0.58, 0.64])
    info_axis = figure.add_axes([0.70, 0.23, 0.27, 0.64])
    info_axis.axis('off')
    radio_axis = figure.add_axes([0.07, 0.03, 0.24, 0.14])
    radio = RadioButtons(radio_axis, ('Choose CURRENT cell',) if recovering else ('Choose START', 'Choose FINISH'), active=0)
    heading_axis = figure.add_axes([0.34, 0.03, 0.22, 0.14])
    heading_axis.set_title('Car facing now' if recovering else 'Start facing', fontsize=11)
    heading_radio = RadioButtons(heading_axis, HEADING_NAMES, active=HEADING_NAMES.index(selection.initial_heading))
    reset_axis = figure.add_axes([0.59, 0.06, 0.18, 0.07])
    reset_button = Button(reset_axis, 'Cancel' if recovering else 'Reset defaults')
    action = {'run_auto': False, 'recovery': None}
    run_button = None
    recovery_button = None
    if robot_mode:
        run_axis = figure.add_axes([0.79, 0.06, 0.18, 0.07])
        run_button = Button(run_axis, 'Continue: Junction solve' if junction_robot else 'Continue: Cell Auto')
    elif junction_preview:
        recovery_axis = figure.add_axes([0.79, 0.06, 0.18, 0.07])
        recovery_button = Button(recovery_axis, 'Confirm position' if recovering else 'Recovery preview [t]')
    figure.suptitle(('Recovery — click the current cell and choose the car direction\n' if recovering else
                    'A* junction route — choose start, finish, and initial direction\n' if junction_preview
                    else 'A* cell route — choose start, finish, and initial direction\n')
                   + ('SIMULATION EXAMPLE — screen testing only.' if simulation_only
                      else 'Junction solver: confirm each A* action. Paused until terminal r.' if junction_robot
                      else 'Legacy cell controller: automatic turns. Paused until terminal r.' if robot_mode
                      else 'Screen preview only; no robot movement.'), fontsize=14)

    def refresh():
        draw_maze(axis, maze, crop)
        axis.set_title('Click the cell where the car is now. Original finish stays fixed.' if recovering
                       else 'Select START or FINISH below, then click a cell')
        path = selection.path
        if path:
            axis.plot([x + 0.5 for x, _ in path], [y + 0.5 for _, y in path],
                      color='royalblue', linewidth=2.5, zorder=3)
        start_offset = 0.27 if selection.initial_heading == 'North' else -0.27
        for cell, label, color, offset in ((selection.start, 'CURRENT' if recovering else 'START', 'darkgreen', start_offset),
                                          (selection.finish, 'FINISH', 'darkred', 0.27)):
            x, y = cell
            axis.text(x + 0.5, y + 0.5 + offset, label, ha='center', va='center', color=color,
                      fontsize=max(6, min(11, 35 / max(maze.rows, maze.cols))), zorder=5,
                      bbox={'facecolor': 'white', 'alpha': 0.9, 'edgecolor': color})
        sx, sy = selection.start
        ax, ay = ((0, -0.30), (0.30, 0), (0, 0.30), (-0.30, 0))[HEADING_NAMES.index(selection.initial_heading)]
        axis.annotate('', xy=(sx + .5 + ax, sy + .5 + ay), xytext=(sx + .5, sy + .5),
                      arrowprops={'arrowstyle': '->', 'color': 'darkorange', 'lw': 2.5}, zorder=6)
        info_axis.clear()
        info_axis.axis('off')
        events = junction_plan(maze, selection.start, selection.finish,
                               heading_side(selection.initial_heading), cell_spacing_m=cell_spacing_m) if junction_preview and path else None
        info = (f'{position_label}: {selection.start}\nFinish: {selection.finish}\n'
                f'{heading_label}: {selection.initial_heading}\n\n'
                '(0,0) is top-left\nx increases right\ny increases down\n\n')
        if path is None:
            info += 'NO OPEN ROUTE\nChoose different cells or\ncheck the logical maze.'
        elif len(path) == 1:
            info += 'Start is already the finish.\n0 cell movements.'
        else:
            shown = path if len(path) <= 12 else path[:8] + ['…'] + path[-2:]
            route = ' → '.join(f'({cell[0]},{cell[1]})' if isinstance(cell, tuple) else cell for cell in shown)
            info += f'A* shortest route\n{len(path) - 1} cell movements\n\n' + '\n'.join(
                textwrap.wrap(route, width=30, break_long_words=False))
        if junction_preview and path:
            shown_events = events if len(events) <= 9 else events[:7] + [None] + events[-1:]
            info = (f'{position_label}: {selection.start}\nFinish: {selection.finish}\n'
                    f'{heading_label}: {selection.initial_heading}\n\n'
                    'Junction plan — expected distances\n'
                    + '\n'.join('…' if event is None else f"{event['cell']}: {event['action']}"
                                + (f" [{CONFIRMATION_KEYS[event['action']]}]" if event['action'] in CONFIRMATION_KEYS else '')
                                + (f" {event['nominal_distance_from_previous_m'] * 100:g}cm"
                                   if event['nominal_distance_from_previous_m'] is not None and event['cells_from_previous'] else '')
                                + (' *' if event['adjacent_openings'] else '')
                                for event in shown_events)
                    + '\n\nFinish: ' + events[-1]['finish_landmark'].replace('_', ' ')
                    + '\nEach junction action will need\nthe shown confirmation key.')
            first = events[0]['action']
            initial = {'FORWARD': 'Forward; already aligned.', 'RIGHT': 'Right 90°, then forward.',
                       'LEFT': 'Left 90°, then forward.', 'U-TURN': 'Turn 180°, then forward.',
                       'STOP': 'Stop; already at finish.'}[first]
            if junction_robot and not FIXED_JUNCTION_TURNS and first in ('LEFT', 'RIGHT'):
                initial = first.title() + ' into the LiDAR-clear passage; align after entry.'
            info += '\n\nFirst: ' + initial
            if any(event['adjacent_openings'] for event in events):
                info += '\n* Adjacent openings: distance check.'
        if recovering:
            info += ('\n\nSelect CURRENT cell, then confirm.' if not selection.cell_clicked
                     else '\n\nCurrent cell selected. Check heading.')
        info += '\n\nBlue line = A* route\nRed edges = WALL\nGreen dashed edges = OPEN\n\n' + (
            'Continue: Junction solve,\nthen press r in the terminal.\nOnly the shown A* key confirms.\nt = manual recovery.' if junction_robot
            else 'Continue: Cell Auto,\nthen press r in the terminal.\nClosing this window cancels.' if robot_mode
            else 'Car near the selected cell center;\nface the chosen direction.\nConfirm position or Cancel.' if recovering else
            'Screen test: Recovery preview [t].' if junction_preview else 'Close the window when done.')
        info += '\nOrange arrow = ' + ('car facing now' if recovering else 'start facing')
        info_axis.text(0, 1, info, transform=info_axis.transAxes, va='top', fontsize=11)
        figure.canvas.draw_idle()
        print(f'\nStart: {selection.start}; finish: {selection.finish}; initial direction: {selection.initial_heading}', flush=True)
        print('A* route: ' + (' → '.join(map(str, path)) if path is not None else 'No open route'), flush=True)
        if path is not None:
            print(f'Cell movements: {len(path) - 1}. Preview only; no robot commands.', flush=True)
            if junction_preview:
                print('Junction instructions (screen review only):', flush=True)
                for event in events:
                    key = CONFIRMATION_KEYS.get(event['action'])
                    print(f"  {event['cell']}: {event['action']}" + (f' [{key}]' if key else '')
                          + f"; approach={event['arrival_heading']}; "
                          f"open sides={','.join(event['open_sides'])}", flush=True)
                    distance = event['nominal_distance_from_previous_m']
                    if event['cells_from_previous']:
                        print('    Nominal map spacing (information only): ' + (f'{distance * 100:g} cm' if distance is not None
                              else 'unknown; cell spacing not supplied')
                              + f" ({event['cells_from_previous']} cell steps from previous landmark).", flush=True)
                    for opening in event['adjacent_openings']:
                        print(f"    Adjacent {opening['relative_side']} openings; divider={opening['divider_state']}. "
                              'Do not count junctions using an OPEN/WALL transition alone.', flush=True)
                print('Finish landmark: ' + events[-1]['finish_landmark'], flush=True)
                if events[-1]['finish_landmark'] == 'not_supported':
                    print('Choose a dead end or corner with a wall ahead in the arrival direction. '
                          'A corridor finish needs additional distance/localization information.', flush=True)

    def click(event):
        if event.inaxes != axis or event.button != 1:
            return
        if figure.canvas.toolbar is not None and figure.canvas.toolbar.mode:
            return
        cell = clicked_cell(maze, event.xdata, event.ydata)
        if cell is None:
            return
        selection.select('start' if recovering or radio.value_selected == 'Choose START' else 'finish', cell)
        refresh()

    def reset(event):
        if recovering:
            plt.close(figure)
            return
        selection.reset()
        radio.set_active(0)
        heading_radio.set_active(HEADING_NAMES.index(selection.initial_heading))

    def select_heading(name):
        selection.select_heading(name)
        refresh()

    def continue_auto(event):
        if simulation_only or selection.path is None:
            print('Physical tests require a real confirmed maze and an OPEN route.', flush=True)
            return
        if junction_robot:
            plan = junction_plan(maze, selection.start, selection.finish, heading_side(selection.initial_heading),
                                 cell_spacing_m=cell_spacing_m)
            if plan[-1]['finish_landmark'] == 'not_supported':
                print('Choose a dead-end finish or a corner with a wall ahead on arrival.', flush=True)
                return
        action['run_auto'] = True
        plt.close(figure)

    def recovery_action(event=None):
        if recovering:
            try:
                result = selection.result()
            except ValueError as error:
                print('Cannot confirm recovery: ' + str(error), flush=True)
                axis.set_title('Cannot confirm: ' + str(error), fontsize=9)
                figure.canvas.draw_idle()
                return
            if on_recovery is not None and on_recovery(result) is False:
                return
            action['recovery'] = result
            print(f"Recovery selection confirmed: cell={result['cell']}, facing={result['heading']}. "
                  'A* replanned; no robot commands.', flush=True)
            plt.close(figure)
            return
        previous = figure._maze_recovery_figure
        if previous is not None and plt.fignum_exists(previous.number):
            previous.show()
            return
        fixed_finish = selection.finish
        def apply_recovery(result):
            if selection.finish != fixed_finish:
                print('Finish changed while recovery was open. Cancel and reopen recovery.', flush=True)
                return False
            selection.select('start', result['cell'])
            heading_radio.set_active(HEADING_NAMES.index(result['heading']))
            return True
        child, _ = build_recovery_window(maze, map_path, fixed_finish, selection.start,
                                         selection.initial_heading, crop, simulation_only=simulation_only,
                                         on_confirm=apply_recovery, cell_spacing_m=cell_spacing_m)
        figure._maze_recovery_figure = child
        child.show()

    def preview_key(event):
        if event.key and event.key.lower() == 't' and junction_preview and not recovering and not robot_mode:
            recovery_action()

    def close_children(event):
        child = figure._maze_recovery_figure
        if child is not None:
            plt.close(child)

    figure.canvas.mpl_connect('button_press_event', click)
    figure.canvas.mpl_connect('key_press_event', preview_key)
    figure.canvas.mpl_connect('close_event', close_children)
    reset_button.on_clicked(reset)
    heading_radio.on_clicked(select_heading)
    if run_button is not None:
        run_button.on_clicked(continue_auto)
    if recovery_button is not None:
        recovery_button.on_clicked(recovery_action)
    # Matplotlib widgets need strong references for callbacks throughout show().
    figure._maze_widgets = (radio, reset_button, run_button)
    figure._maze_heading_widget = heading_radio
    figure._maze_recovery_widget = recovery_button
    figure._maze_recovery_figure = None
    figure._maze_action = action
    refresh()
    return figure, selection


def build_recovery_window(maze, map_path, finish, current_cell, current_heading, crop=None,
                          *, simulation_only=False, on_confirm=None, cell_spacing_m=None):
    """Build a stopped recovery selection UI; caller owns any eventual motion."""
    return build_window(maze, map_path, crop, simulation_only=simulation_only, junction_preview=True,
                        recovery_from=(current_cell, current_heading, finish), on_recovery=on_confirm,
                        cell_spacing_m=cell_spacing_m)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logical_maze_json', nargs='?', default=LOGICAL_MAZE_JSON,
                        help='Confirmed Stage 2 JSON; omit to choose it in a file picker.')
    parser.add_argument('--robot', action='store_true', help='Drive the real junction solver after GUI selection and terminal r.')
    parser.add_argument('--junction-preview', action='store_true',
                        help='Review A* junction instructions on screen only; no robot controller.')
    parser.add_argument('--cell-mode', action='store_true',
                        help='Use the older cell-distance method, with automatic turns in robot mode.')
    args = parser.parse_args()
    if args.junction_preview and args.cell_mode:
        parser.error('Choose the junction preview or --cell-mode, not both.')
    if args.robot and args.junction_preview:
        parser.error('--junction-preview is screen-only. Use --robot without that flag for real movement.')
    try:
        if args.logical_maze_json is None:
            print('Choose a confirmed logical_maze.json or a simulation example in the file picker.', flush=True)
            args.logical_maze_json = choose_logical_maze_file()
            if args.logical_maze_json is None:
                print('Cancelled. No logical maze selected.')
                return
        path = Path(args.logical_maze_json).expanduser().resolve()
        maze, record = load_logical_maze(path, allow_simulation=not args.robot)
        cell_spacing_m = preview_cell_spacing(record)
        EndpointSelection(maze, DEFAULT_START, DEFAULT_FINISH, INITIAL_HEADING)
        if args.robot:
            from robot_auto import prepare_auto_configuration
            config, _ = prepare_auto_configuration(maze, record, DEFAULT_START, CELL_EAST_WEST_CM, CELL_NORTH_SOUTH_CM,
                                                  movement_settings=solver_controller_settings())
            if not args.cell_mode:
                solver_junction_settings().validate(config)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    crop = None
    image_path = path.parent / record.get('source', {}).get('crop_image', 'maze_crop.png')
    if image_path.is_file():
        with Image.open(image_path) as image:
            crop = np.array(image.convert('L'))
    simulation_only = record.get('simulation_only') is True
    print(('Simulation example: ' if simulation_only else 'Confirmed logical maze: ') + str(path) + '\n'
          f'Size: {maze.cols} columns × {maze.rows} rows. Mouse selection enabled.\n'
          + ('This generated example does not describe your physical maze.\n' if simulation_only
             else 'Choose the direction the car will face at the selected start.\n') +
          ('Robot movement requested. Choose cells, continue, then press r in the terminal.\n'
           if args.robot else 'Screen preview only; no robot commands.'), flush=True)
    import matplotlib.pyplot as plt
    figure, selection = build_window(maze, path, crop, simulation_only=simulation_only, robot_mode=args.robot,
                                    junction_preview=not args.cell_mode, cell_spacing_m=cell_spacing_m)
    plt.show()
    plt.close(figure)
    print(f'Final selection: start={selection.start}, finish={selection.finish}, heading={selection.initial_heading}.', flush=True)
    if args.robot:
        if not figure._maze_action['run_auto']:
            print('Robot Auto cancelled. No movement controller started.', flush=True)
            return
        from robot_auto import prepare_auto_configuration, run_auto_robot
        try:
            config, settings = prepare_auto_configuration(maze, record, selection.start,
                                                         CELL_EAST_WEST_CM, CELL_NORTH_SOUTH_CM,
                                                         movement_settings=solver_controller_settings())
            if args.cell_mode:
                run_auto_robot(maze, selection.finish, config, settings, initial_heading=selection.initial_heading)
            else:
                from robot_junction import run_junction_robot
                run_junction_robot(maze, selection.finish, config, settings, initial_heading=selection.initial_heading,
                                   junction_settings=solver_junction_settings(), map_path=path, crop=crop)
        except ValueError as error:
            parser.error(str(error))


if __name__ == '__main__':
    main()
