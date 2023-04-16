def l2_distance(state1, state2):
  running_sum = 0
  for param in state1.keys():
    if 'mean' in param or 'var' in param or 'batches_tracked' in param:
      continue
    running_sum += (state1[param] - state2[param]).pow(2).sum()
  return running_sum.sqrt()

def norm(state):
  running_sum = 0
  for param in state.keys():
    if 'mean' in param or 'var' in param or 'batches_tracked' in param:
      continue
    running_sum += (state[param]).pow(2).sum()
  return running_sum.sqrt()