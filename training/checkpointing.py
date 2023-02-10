from foundations import paths
from foundations.step import Step
from training.metric_logger import MetricLogger
from environment import environment

def save_checkpoint_callback(output_location, step, model, optimizer, scheduler, logger):
    environment.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': None if scheduler is None else scheduler.state_dict(),
        'ep': step.ep,
        'it': step.it,
        'logger': str(logger)
    }, paths.checkpoint(output_location))

def restore_checkpoint(output_location, model, optimizer, scheduler, iterations_per_epoch):
    checkpoint_location = paths.checkpoint(output_location)
    if not environment.exists(checkpoint_location):
        print('not using checkpoint')
        return None, None
    checkpoint = environment.load(checkpoint_location)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    step = Step.from_epoch(checkpoint['ep'], checkpoint['it'], iterations_per_epoch)
    logger = MetricLogger.create_from_str(checkpoint['logger'])
    print(f'using check at step {step.ep_it_str}')
    return step, logger