"""

Compatible with the dataset here:
/eos/home-j/jkiesele/ML4Reco/Gun20Part_NewMerge/train

On flatiron:
/mnt/ceph/users/jkieseler/HGCalML_data/Gun20Part_NewMerge/train

not compatible with datasets before end of Jan 2022

"""
import wandb


import tensorflow as tf

from tensorflow.keras.layers import Dense, Concatenate

from DeepJetCore.DJCLayers import StopGradient

from Layers import RaggedGlobalExchange, DistanceWeightedMessagePassing, DictModel
from Layers import RaggedGravNet, ScaledGooeyBatchNorm2
from Regularizers import AverageDistanceRegularizer
from LossLayers import LLBasicObjectCondensation_nod
from wandb_callback import wandbCallback
from DebugLayers import PlotCoordinates

from model_blocks import (
    condition_input,
    extent_coords_if_needed,
    create_outputs_condensation_only,
    re_integrate_to_full_hits_clusteronly,
)

from callbacks import plotClusterSummary
from argparse import ArgumentParser
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback


parser = ArgumentParser("Run the training")
# parser.add_argument("input_dataset")
# parser.add_argument("output_dir")
parser.add_argument(
    "--interactive", help="prints output to screen", default=False, action="store_true"
)
parser.add_argument(
    "--run_name", "-name", help="wandb run name", default="HGCalML-1 baseline training"
)
parser.add_argument("--epochs", "-e", help="wandb run name", default=500, type=int)
parser.add_argument("--nbatch", "-b", help="batch size", default=10000, type=int)
# parser.add_argument('inputDataCollection')
# parser.add_argument('outputDir')

# loss options:
loss_options = {
    # here and in the following energy = momentum
    # "energy_loss_weight": 1.0,
    "q_min": 0.5,
    # addition to original OC, adds average position for clusterin
    # usually 0.5 is a reasonable value to break degeneracies
    # and keep training smooth enough
    "use_average_cc_pos": 0.5,
    # "classification_loss_weight": 0.0,
    # "position_loss_weight": 0.0,
    # "timing_loss_weight": 0.0,
    "beta_loss_scale": 1.0,
    # these weights will downweight low energies, for a
    # training sample with a good energy distribution,
    # this won't be needed.
    # "use_energy_weights": False,
    # this is the standard repulsive hinge loss from the paper
    "implementation": "hinge",
}


# elu behaves well, likely fine
dense_activation = "elu"

# record internal metrics every N batches
record_frequency = 10
# plot every M times, metrics were recorded. In other words,
# plotting will happen every M*N batches
plotfrequency = 50

learningrate = 1e-4

# this is the maximum number of hits (points) per batch,
# not the number of events (samples). This is safer w.r.t.
# memory
# args = parser.parse_args()


# iterations of gravnet blocks
n_neighbours = [64, 64]

# 3 is a bit low but nice in the beginning since it can be plotted
n_cluster_space_coordinates = 3
n_gravnet_dims = 3


def gravnet_model(
    Inputs,
    td,
    debug_outdir=None,
    plot_debug_every=record_frequency * plotfrequency,
):
    ####################################################################################
    ##################### Input processing, no need to change much here ################
    ####################################################################################

    input_list = td.interpretAllModelInputs(Inputs, returndict=True)
    print("Output of interpretAllModelInputs:\n", input_list)
    input_list = condition_input(input_list, no_scaling=True)

    # just for info what's available, prints once
    print("available inputs", [k for k in input_list.keys()])

    rs = input_list["row_splits"]
    t_idx = input_list["t_idx"]
    energy = input_list["rechit_energy"]
    c_coords = input_list["coords"]

    ## build inputs
    #print("INPUTS SHAPE", input_list["coords"].shape, input_list["features"].shape)
    x_in = Concatenate()([input_list["coords"], input_list["features"]])

    x_in = ScaledGooeyBatchNorm2(
        fluidity_decay=0.1  # freeze out quickly, just to get good input preprocessing
    )(x_in)

    x = x_in

    c_coords = ScaledGooeyBatchNorm2(fluidity_decay=0.1)(c_coords)  # same here

    ####################################################################################
    ##################### now the actual model goes below ##############################
    ####################################################################################

    # output of each iteration will be concatenated
    allfeat = []

    # extend coordinates already here if needed, just as a good starting point
    c_coords = extent_coords_if_needed(c_coords, x, n_gravnet_dims)

    for i in range(len(n_neighbours)):

        # derive new coordinates for clustering
        x = RaggedGlobalExchange()([x, rs])

        x = Dense(64, activation=dense_activation)(x)
        x = Dense(64, activation=dense_activation)(x)
        x = Dense(64, activation=dense_activation)(x)
        x = Concatenate()([c_coords, x])  # give a good starting point
        x = ScaledGooeyBatchNorm2()(x)

        xgn, gncoords, gnnidx, gndist = RaggedGravNet(
            n_neighbours=n_neighbours[i],
            n_dimensions=n_gravnet_dims,
            n_propagate=64,  # this is the number of features that are exchanged
            n_filters=64,  # output dense
            feature_activation="elu",
        )([x, rs])

        x = Concatenate()([x, xgn])

        # mostly to record average distances etc. can be used to force coordinates
        # to be within reasonable range (but usually not needed)
        gndist = AverageDistanceRegularizer(strength=1e-6, record_metrics=True)(gndist)

        # for information / debugging, can also be safely removed
        gncoords = PlotCoordinates(
            plot_every=plot_debug_every, outdir=debug_outdir, name="gn_coords_" + str(i)
        )([gncoords, energy, t_idx, rs])
        # we have to pass them downwards, otherwise the layer above gets optimised away
        # but we don't want the gradient to be disturbed, so it gets stopped
        gncoords = StopGradient()(gncoords)
        x = Concatenate()([gncoords, x])

        # this repeats the distance weighted message passing step from gravnet
        # on the same graph topology
        x = DistanceWeightedMessagePassing([64, 64], activation=dense_activation)(
            [x, gnnidx, gndist]
        )

        x = ScaledGooeyBatchNorm2()(x)

        x = Dense(64, activation=dense_activation)(x)
        x = Dense(64, activation=dense_activation)(x)
        x = Dense(64, activation=dense_activation)(x)

        x = ScaledGooeyBatchNorm2()(x)

        allfeat.append(x)

    x = Concatenate()([c_coords] + allfeat)  # gives a prior to the clustering coords
    # create one global feature vector
    xg = Dense(512, activation=dense_activation, name="glob_dense_" + str(i))(x)
    x = RaggedGlobalExchange()([xg, rs])
    x = Concatenate()([x, xg])
    # last part of network
    x = Dense(64, activation=dense_activation)(x)
    x = ScaledGooeyBatchNorm2()(x)
    x = Dense(64, activation=dense_activation)(x)
    x = ScaledGooeyBatchNorm2()(x)
    x = Dense(64, activation=dense_activation)(x)
    x = ScaledGooeyBatchNorm2()(x)

    #######################################################################
    ########### the part below should remain almost unchanged #############
    ########### of course with the exception of the OC loss   #############
    ########### weights                                       #############
    #######################################################################

    # use a standard batch norm at the last stage

    (pred_beta, pred_ccoords) = create_outputs_condensation_only(
        x, n_ccoords=n_cluster_space_coordinates
    )

    # loss
    pred_beta = LLBasicObjectCondensation_nod(
        scale=1.0,
        record_metrics=True,
        print_loss=True,
        name="BasicOCLoss",
        **loss_options
    )(  # oc output and payload
        [
            pred_beta,
            pred_ccoords,
        ]
        + [input_list["t_idx"], input_list["row_splits"]]
    )

    # fast feedback
    pred_ccoords = PlotCoordinates(
        plot_every=plot_debug_every, outdir=debug_outdir, name="condensation_coords"
    )([pred_ccoords, pred_beta, input_list["t_idx"], rs])

    # # just to have a defined output, only adds names
    model_outputs = re_integrate_to_full_hits_clusteronly(
        input_list, pred_ccoords, pred_beta
    )

    return DictModel(inputs=Inputs, outputs=Inputs)


import training_base_hgcal

train = training_base_hgcal.HGCalTraining(parser=parser)

args = train.args
wandb.init(
    project="hgcalml-1",
    tags=["debug", "small_dataset"],
    name=args.run_name,
    entity="fcc_",
)
wandb.run.log_code(".")
wandb.config["args"] = vars(args)
nbatch = args.nbatch


if not train.modelSet():
    train.setModel(
        gravnet_model,
        td=train.train_data.dataclass(),
        debug_outdir=train.outputDir + "/intplots",
    )

    train.setCustomOptimizer(tf.keras.optimizers.Nadam(clipnorm=1.0, epsilon=1e-2))
    #
    train.compileModel(learningrate=1e-4)

    train.keras_model.summary()
    #wandb.watch(train.keras_model, log="all", log_freq=100)


verbosity = 2
import os

publishpath = (
    None  # this can be an ssh reachable path (be careful: needs tokens / keypairs)
)

# establish callbacks

"""
simpleMetricsCallback(
    output_file=train.outputDir+'/metrics.html',
    record_frequency= record_frequency,
    plot_frequency = plotfrequency,
    select_metrics='FullOCLoss_*loss',
    publish=publishpath #no additional directory here (scp cannot create one)
    ),

simpleMetricsCallback(
    output_file=train.outputDir+'/latent_space_metrics.html',
    record_frequency= record_frequency,
    plot_frequency = plotfrequency,
    select_metrics='average_distance_*',
    publish=publishpath
    ),


simpleMetricsCallback(
    output_file=train.outputDir+'/val_metrics.html',
    call_on_epoch=True,
    select_metrics='val_*',
    publish=publishpath #no additional directory here (scp cannot create one)
    ),
"""
cb = [wandbCallback()]


cb += [
    plotClusterSummary(
        outputfile=train.outputDir + "/clustering/",
        samplefile=train.val_data.getSamplePath(train.val_data.samples[0]),
        after_n_batches=200,
        log_wandb=True,
        on_epoch_end=True,
    )
]

# cb=[]


train.change_learning_rate(learningrate)

model, history = train.trainModel(nepochs=3, batchsize=nbatch, additional_callbacks=cb)

print("freeze BN")
# Note the submodel here its not just train.keras_model
# for l in train.keras_model.layers:
#    if 'FullOCLoss' in l.name:
#        l.q_min/=2.

train.change_learning_rate(learningrate / 2.0)

model, history = train.trainModel(
    nepochs=args.epochs, batchsize=nbatch, additional_callbacks=cb
)
