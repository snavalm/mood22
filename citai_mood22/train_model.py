
import os

from tqdm import tqdm

from monai.metrics import DiceMetric
from torchmetrics import AveragePrecision

from monai.inferers import sliding_window_inference

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
### Using torch.utils.data.DataLoader instead of monai because monai is giving me a memory error in some environments

from utils import default,  configure_logging, add_common_args
import preprocessing

import pandas as pd

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def prepare_data(config):
    train_ds = preprocessing.get_dataset(image_only=True, **config.train_dataset_config, )
    train_loader = DataLoader(train_ds, **default(config,"train_loader_config"))

    if hasattr(config,"validation_dataset_config"):
        val_ds = preprocessing.get_dataset(image_only = False, **config.validation_dataset_config)
        val_loader = DataLoader(val_ds, **default(config,"val_loader_config"))
    else:
        val_loader = None

    return train_loader, val_loader


def validation(val_loader, test_run = False):
    epoch_iterator_val = tqdm(
        val_loader, desc = "Validate (X / X Steps) (dice=X.X)", dynamic_ncols = True, position = 0, leave = True )

    model.eval()
    types = []
    samples_predictions = []
    samples_labels = []

    ap_metric.reset()
    dice_metric.reset()

    with torch.no_grad():
        for step, batch in enumerate( epoch_iterator_val ):
            val_inputs, val_labels = batch["image"].to( device ), batch["label"].to( device )

            val_outputs = sliding_window_inference( val_inputs, roi_size = img_size,
                                                    sw_batch_size = 2,
                                                    predictor = model.predict,)

            dice_metric( y_pred = ( val_outputs > 0  ).float(), y = val_labels )

            # For AP just keep 1% of random voxels
            n_voxels = val_outputs[0].nelement()//100
            idx = torch.randint(val_outputs[0].nelement(),size=(n_voxels,))
            samples_predictions.append(torch.cat([o.flatten()[idx] for o in val_outputs]))
            samples_labels.append(torch.cat([o.flatten()[idx] for o in val_labels]))

            epoch_iterator_val.set_description( "Validate %d " % (global_step))

            types.extend( batch['type'] )

            if test_run and (step >=2):
                return None

        # Calculate metrics
        samples_predictions = torch.cat(samples_predictions)
        samples_labels = torch.cat(samples_labels)
        ap_val = ap_metric(samples_predictions,
                           samples_labels)
        dice_val = dice_metric.aggregate()

        # Calculate the mean DICE by type of anomaly and log
        df = pd.DataFrame( dice_val.cpu().numpy(), columns = ['DICE'] )
        df['types'] = types

        dice_summary = df.pivot_table( index = 'types', values = ['DICE'], aggfunc = 'mean' )

        mean_val_dice = dice_val.mean().item()

        if wandb_config is not None:
            for t, v in dice_summary.iterrows():
                wandb.log({"DICE_{}".format(t): v[0].item()}, step=global_step)

            wandb.log( {"DICE_mean": mean_val_dice}, step = global_step )
            wandb.log( {"AP_mean": ap_val.item()} , step = global_step )

    # Set the model to train after
    model.train()

    return ap_val.item()

def train(train_loader):

    global global_step
    global relative_step
    global best_val_measure
    global loss_ema

    model.train()

    epoch_iterator = tqdm(
        train_loader, desc = "Training (X / X Steps) (loss=X.X)", dynamic_ncols = True,
        position = 0, leave = True
    )
    for batch in epoch_iterator:

        optimizer.zero_grad()

        with autocast():
            loss, pred = model(  batch["image"].to(device) , batch["mask"].to(device) )

        amp_grad_scaler.scale( loss ).backward()
        amp_grad_scaler.step( optimizer )
        amp_grad_scaler.update()
        scheduler.step()

        if loss.isnan() or loss.isinf():
            raise ValueError("Loss is {} in step {}".format(loss.item(),global_step))

        loss_ema = loss.item() if loss_ema is None else (loss_ema * .95) + (loss.item() * .05)

        epoch_iterator.set_description( "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss_ema))

        if global_step % 100 == 0:
            if wandb_config is not None:
                wandb.log( {"train_loss": loss_ema}, step = global_step )
                wandb.log( {"lr": optimizer.param_groups[0]['lr']}, step = global_step)

        global_step += 1

        if (val_loader is not None) and ((global_step % eval_frequency == 0) or (global_step == max_iterations)):

            current_val_measure = validation( val_loader )

            if current_val_measure > best_val_measure:
                best_val_measure = current_val_measure
                torch.save({"step":global_step, "val_measure":current_val_measure, "state_dict":model.state_dict()},
                            os.path.join( checkpoint_dir, "best_metric_model.pth" ) )
                print( "Model Was Saved ! Current Best Val: {} Current Val: {}".format( current_val_measure, current_val_measure ) )
            else:
                print( "Model Was Not Saved ! Current Best Val: {} Current Val: {}".format( best_val_measure, current_val_measure ) )

        if global_step == max_iterations:
            break

    if wandb_config is not None:
        wandb.log({"lr":optimizer.param_groups[0]['lr']}, step=global_step)




if __name__ == "__main__":
    import argparse
    import importlib

    arg_parser = argparse.ArgumentParser( description = "Train models" )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest = "experiment_config",
        required = True,
        help = "Experiment specification file .json",
    )
    add_common_args( arg_parser )
    args = arg_parser.parse_args()
    configure_logging( args )

    config = importlib.import_module(args.experiment_config)
    experiment_id = config.experiment_id

    checkpoint_dir = os.path.join(config.checkpoint_dir, experiment_id)
    if os.path.exists(checkpoint_dir):
        run_id = 1
        while os.path.exists(checkpoint_dir + '_run{:03d}'.format(run_id)):
            run_id += 1
        checkpoint_dir = checkpoint_dir + '_run{:03d}'.format(run_id)
        experiment_id = experiment_id + '_run{:03d}'.format(run_id)

    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb_config = default(config,'wandb_config')

    if wandb_config is not None:
        try:
            import wandb
            wandb.init(name=experiment_id, **wandb_config)
        except:
            raise ImportError("Install wandb or comment wandb_config in the experiment config file")

    train_loader, val_loader = prepare_data(config)

    img_size = config.patch_size
    model = default(config,'model')

    device = torch.device( "cuda:{}".format(config.cuda_devices[0]) if torch.cuda.is_available() else "cpu" )
    model = model.to( device )

    torch.backends.cudnn.benchmark = True

    optimizer = default(config,'optimizer')(model.parameters())
    scheduler = default(config,'scheduler')(optimizer)

    amp_grad_scaler = GradScaler()

    max_iterations = default(config,'max_iterations')
    eval_frequency = default(config,'eval_frequency')

    global_step = config.step if hasattr(config,'step') else 0
    best_val_measure = config.best_val_measure if hasattr(config,'best_val_measure') else 0
    loss_ema = None

    ap_metric = AveragePrecision( num_classes = 1 )
    dice_metric = DiceMetric( include_background = False, reduction = "none", ignore_empty = False )

    # Start with a validation to check it works fine
    if val_loader is not None:
        validation(val_loader, test_run = True)

    # Training
    while global_step < max_iterations:
        train( train_loader)

    torch.save({"step": global_step, "state_dict": model.state_dict()},
               os.path.join(checkpoint_dir, "final_model.pth"))


