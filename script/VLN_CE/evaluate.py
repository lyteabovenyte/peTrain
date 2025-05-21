import numpy as np

def compute_metrics(trajectories):
    """
    trajectories: list of dicts, each with:
      'final_pos': [x,y,z] final agent position,
      'path_length': float (meters),
      'episode': {
         'goal_pos': [x,y,z], 'goal_radius': float, 'geodesic_distance': float
      }
    """
    success_list = []
    spl_list = []
    for traj in trajectories:
        goal_pos = np.array(traj['episode']['goal_pos'])
        goal_rad = traj['episode']['goal_radius']
        final_pos = np.array(traj['final_pos'])
        dist_to_goal = np.linalg.norm(final_pos - goal_pos)
        success = dist_to_goal <= goal_rad  # success if within radius
        success_list.append(float(success))
        L = traj['episode']['geodesic_distance']  # shortest-path length
        P = traj['path_length']
        if success and P > 0:
            spl = L / max(P, L)
        else:
            spl = 0.0
        spl_list.append(spl)
    success_rate = np.mean(success_list)
    SPL = np.mean(spl_list)
    return success_rate, SPL