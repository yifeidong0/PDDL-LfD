# -----------------------------------------------------------------------------
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of the LogicLfD project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yan Zhang <yan.zhang@idiap.ch>
# -----------------------------------------------------------------------------

"""
This scripts illustrates how Logic-LfD is used to solve 
the block stacking problem with different initial states
"""
import os
import time
import pickle
import numpy as np
from config import *
import pybullet as p
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import print_solution, PDDLProblem
from pddlstream.language.generator import from_gen_fn
from pddlstream.utils import read, INF

from examples.pybullet.utils.pybullet_tools.utils import \
    LockRenderer, HideOutput

from experiments.hooks_world.primitives import \
    get_grasp_gen, get_stable_gen, get_stack_gen, get_hook_place_gen

from experiments.hooks_world.env_hook_tools import HookWorld

def pddlstream_from_problem(robots_info, tables_info, target_objects_info,
                            init, goal=None):
    
	domain_path = absjoin(EXP_PATH, 'hooks_world/domain.pddl')
	stream_path = absjoin(EXP_PATH, 'hooks_world/stream.pddl')
	domain_pddl = read(domain_path)
	stream_pddl = read(stream_path)
	constant_map = {}

	robot_ids = [robot_info["id"] for name, robot_info in robots_info.items()]
	table_ids = [table_info["id"] for name, table_info in tables_info.items()]
	block_ids = [block_info["id"] for name, block_info in target_objects_info.items()]
	start_id = min(block_ids)
	end_id = max(block_ids)
	num_target_objects = len(block_ids)

	table_id = table_ids[0]
	if goal is None:
		# TODO: set the goal of TAMP problem here
		goal = ("and", *[("on-block", i, i + 1) for i in range(start_id, num_target_objects + 1)],
						("on-table", num_target_objects + 1, table_id),)
		# print("Template problem goal: ", goal)

	stream_map = {
		"find-grasp": from_gen_fn(get_grasp_gen(robot_ids[0])),
		"find-table-place": from_gen_fn(get_stable_gen(fixed=table_ids)),
		"find-block-place": from_gen_fn(get_stack_gen()),
		"find-hook-place": from_gen_fn(get_hook_place_gen()),
	}

	return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

def solve_template_tamp_problem(robots_info, tables_info, 
                                target_objects_info, hooks_info, goal=None):
	"""
	This function generate a plan for solving the template TAMP problem
	"""
	with HideOutput():
		env = HookWorld(use_gui=USE_GUI)
		init_table_states = env.load_table(tables_info)
		init_robot_states = env.load_robot(robots_info)
		init_block_states = env.template_load_objects(target_objects_info)
		init_hook_states = env.load_hooks(hooks_info) # must be after loading objects and others
		init = init_robot_states + init_table_states + init_block_states + init_hook_states

	statics, fluents = env.get_logical_state(robots_info=robots_info,
										  tables_info=tables_info,
										  target_objects_info=target_objects_info)
	print("Initial logical state")
	print("############Initial state: ", init)
	print("############statics: ", statics)
	print("############fluents: ", fluents)
	init += statics + fluents
	problem = pddlstream_from_problem(robots_info, tables_info, 
										target_objects_info, init, goal=goal)
	if VERBOSE:
		print("Template problem init: ", problem.init)
		print("Template problem goal: ", problem.goal)

	with LockRenderer(lock=False):
		init_t = time.time()
		solution = solve(problem, success_cost=INF, unit_costs=True,
						debug=DEBUG, verbose=VERBOSE)
		cost_t = time.time() - init_t
	
	# _, _, _ = print_solution(solution)
	print(f"\n Time cost for solving the template problem: {cost_t:.3f} s")

	plan, _, _ = solution
	print("############Plan: ", plan)
	if plan is None:
		env.disconnect()

	# Visualization
	if USE_GUI:
		# Visualization
		env.reset(robots_info=robots_info, target_objects_info=target_objects_info)
		env.postprocess_plan(plan)
		time.sleep(1)
	env.disconnect()

	return plan

 
if __name__ == "__main__":
	"""Generate the action plan for template problem"""
	DEBUG = False; VERBOSE = False # By default, not log info
	USE_GUI = True; np.random.seed(0)

	# set robots info 
	robots_info = {"panda": {
			"urdf": PANDA_URDF,
			"conf": [0, 0, 0, -1.5, 0, 1.5, 0.717, 0.06, 0.06]}}

	# set tables info
	tables_info = {"floor": {
			"urdf": absjoin(OBJECT_URDF, 'short_floor.urdf'),
			"pose": [0.0, 0.0, 0.0]}}

	# set objects info
	target_objects_info = {}
	target_objects_info["tape"] = {"urdf": TAPE_URDF}
	target_objects_info["scissors"] = {"urdf": SCISSORS_URDF}

	# set hooks info
	hooks_info = {}
	hooks_info["hook-coat"] = {"urdf": HOOK_COAT_URDF, "pose": ([0.2, 0.5, 0.5], p.getQuaternionFromEuler([1.57, 0, 0]))}
	hooks_info["hook-slatwall"] = {"urdf": HOOK_SLATWALL_URDF, "pose": ([0.5, 0.5, 0.5], p.getQuaternionFromEuler([1.57, 0, 0]))}

	init_plan = solve_template_tamp_problem(robots_info, tables_info,
											target_objects_info, hooks_info, goal=None)
