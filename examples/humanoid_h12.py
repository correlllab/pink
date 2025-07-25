import time
import pink
import random
import qpsolvers
import meshcat
import meshcat_shapes

import numpy as np
import pinocchio as pin

from pinocchio.visualize import MeshcatVisualizer

def inspect_model(model):
    print(f'nq (configuration dim) = {model.nq}')
    print(f'nv (velocity dim)      = {model.nv}')
    print(f'nbodies                = {model.nbodies}')
    print(f'njoints                = {model.njoints}')

    # iterate through bodies
    print('H1_2 Robot Bodies')
    print('=================')
    for frame in model.frames:
        if frame.type == pin.FrameType.BODY:
            print(f'Body {frame.name}')
            print(f'Parent Joint: {model.names[frame.parentJoint]}')
            print()

    # iterate through joints
    print('H1_2 Robot Joints')
    print('=================')
    for j_idx, joint in enumerate(model.joints):
        # Note: joint[0] is the universe/root; skip if you like
        print(f'''Joint {j_idx}: {model.names[j_idx]}
            idx_q: {joint.idx_q}, idx_v: {joint.idx_v}
            ''')

    print('H1_2 Joint Limits')
    print('=================')
    print('Upper Position Limit:', model.upperPositionLimit)
    print('Lower Position Limit:', model.lowerPositionLimit)

    print('H1_2 Velocity Limits')
    print('====================')
    print('Velocity Limit:', model.velocityLimit)

def main(dt=0.02):
    # load h_2 urdf
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        filename='./assets/h1_2/h1_2.urdf',
        package_dirs='./assets/h1_2'
    )
    data = model.createData()

    # print basic data
    inspect_model(model)

    # load visualizer
    try:
        viz = MeshcatVisualizer(model, collision_model, visual_model,
                                copy_models=False, data=data)
        viz.initViewer(open=True)
        viz.loadViewerModel('unitree_h1_2')
    except ImportError as err:
        print('ImportError: MeshcatVisualizer requires the meshcat package.')
        print(err)
        exit(0)
    # show frames
    viewer = viz.viewer
    meshcat_shapes.frame(viewer['end_effector_target'], opacity=0.5)
    meshcat_shapes.frame(viewer['end_effector'], opacity=1.0)

    # set up configuration
    q = pin.neutral(model)
    configuration = pink.Configuration(model, data, q)

    # set up tasks
    end_effector_task = pink.FrameTask(
        'left_wrist_yaw_link',
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=1.0,  # tuned for this setup
    )
    posture_task = pink.PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )
    tasks = [end_effector_task, posture_task]
    for task in tasks:
        task.set_target_from_configuration(configuration)

    # set up limits
    limits = [
        pink.limits.ConfigurationLimit(model),
        pink.limits.VelocityLimit(model)
    ]

    # choose QP solver
    solver = qpsolvers.available_solvers[0]
    if 'daqp' in qpsolvers.available_solvers:
        solver = 'daqp'

    # main loop
    while True:
        # reset configuration to neutral
        configuration.q = pin.neutral(model)
        viz.display(configuration.q)
        # sample a random target position
        input('Press enter to randomly sample a target')
        target_pos = [
            random.uniform(0.2, 0.5),
            random.uniform(0.0, 0.5),
            random.uniform(0.0, 0.5)
        ]
        # update task targets
        end_effector_target = end_effector_task.transform_target_to_world
        end_effector_target.translation[0] = target_pos[0]
        end_effector_target.translation[1] = target_pos[1]
        end_effector_target.translation[2] = target_pos[2]

        start_time = time.time()
        while time.time() - start_time < 10.0:
            frame_start_time = time.time()
            # update visualization frames
            viewer['end_effector_target'].set_transform(end_effector_target.np)
            viewer['end_effector'].set_transform(
                configuration.get_transform_frame_to_world(
                    end_effector_task.frame
                ).np
            )
            # solve IK
            vel = pink.solve_ik(
                configuration,
                tasks,
                dt=dt,
                limits=limits,
                solver=solver
            )
            # update configuration
            configuration.integrate_inplace(vel, dt)
            viz.display(configuration.q)

            # early break
            if np.linalg.norm(end_effector_task.compute_error(configuration)) < 1e-3:
                print('Target reached!')
                break

            time.sleep(max(0.0, dt - (time.time() - frame_start_time)))

        input('Press enter to continue to next target')

if __name__ == '__main__':
    main(dt=0.02)
