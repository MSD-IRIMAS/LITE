from utils.utils import load_data, znormalisation, encode_labels, create_directory
import sys
import os
import pandas as pd
import numpy as np
import json
import argparse
from distutils.util import strtobool

from classifiers.lite import LITE
from classifiers.litemv import LITEMV

from sklearn.metrics import accuracy_score


def get_args():
    parser = argparse.ArgumentParser(
        description="Choose to apply which classifier on which dataset with number of runs."
    )

    parser.add_argument(
        "--dataset",
        help="which dataset to run the experiment on.",
        type=str,
        default="Coffee",
    )

    parser.add_argument(
        "--classifier",
        help="which classifier to use",
        type=str,
        choices=["LITE","LITEMV"],
        default="LITE",
    )

    parser.add_argument("--runs", help="number of runs to do", type=int, default=5)

    parser.add_argument(
        "--output-directory",
        help="output directory parent",
        type=str,
        default="results/",
    )

    parser.add_argument(
        "--track-emissions", type=lambda x: bool(strtobool(x)), default=True
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    output_directory_parent = args.output_directory
    create_directory(output_directory_parent)
    output_directory_parent = output_directory_parent + args.classifier + "/"
    create_directory(output_directory_parent)

    xtrain, ytrain, xtest, ytest = load_data(file_name=args.dataset)

    length_TS = int(xtrain.shape[1])

    xtrain = znormalisation(xtrain)
    xtrain = np.expand_dims(xtrain, axis=2)

    xtest = znormalisation(xtest)
    xtest = np.expand_dims(xtest, axis=2)

    ytrain = encode_labels(ytrain)
    ytest = encode_labels(ytest)

    if os.path.exists(output_directory_parent + "results_ucr.csv"):
        df = pd.read_csv(output_directory_parent + "results_ucr.csv")

        file_names = list(df["dataset"])
        if args.dataset in file_names:
            exit()
    else:
        if args.track_emissions:
            df = pd.DataFrame(
                columns=[
                    "dataset",
                    args.classifier + "-mean",
                    args.classifier + "-std",
                    args.classifier + "Time",
                    "Training Time-mean",
                    "Testing Time-mean",
                    "CO2 Consumption-mean",
                    "Energy Consumption-mean",
                    "Country",
                    "Region",
                ]
            )
        else:
            df = pd.DataFrame(
                columns=[
                    "dataset",
                    args.classifier + "-mean",
                    args.classifier + "-std",
                    args.classifier + "Time",
                ]
            )

    ypred = np.zeros(shape=(len(ytest), len(np.unique(ytest))))

    Scores = []

    if args.track_emissions:
        training_time = []
        testing_time = []

        co2_consumption = []
        energy_consumption = []

    for _run in range(args.runs):
        output_directory = output_directory_parent + "run_" + str(_run) + "/"
        create_directory(output_directory)
        output_directory = output_directory + args.dataset + "/"
        create_directory(output_directory)

        if args.classifier == "LITE":
            clf = LITE(
                output_directory=output_directory,
                length_TS=length_TS,
                n_classes=len(np.unique(ytrain)),
            )
        elif args.classifier == "LITEMV":
            clf = LITEMV(
                output_directory=output_directory,
                length_TS=length_TS,
                n_classes=len(np.unique(ytrain)),
            )
        else:
            raise ValueError("Choose an existing classifier.")

        if not os.path.exists(output_directory + "loss.pdf"):
            if args.track_emissions:
                dict_emissions = clf.fit_and_track_emissions(
                    xtrain=xtrain, ytrain=ytrain, xval=xtest, yval=ytest, plot_test=True
                )
            else:
                clf.fit(
                    xtrain=xtrain, ytrain=ytrain, xval=xtest, yval=ytest, plot_test=True
                )

        else:
            if args.track_emissions:
                with open(output_directory + "dict_emissions.json") as json_file:
                    dict_emissions = json.load(json_file)

        if args.track_emissions:
            co2_consumption.append(dict_emissions["co2"])
            energy_consumption.append(dict_emissions["energy"])

            training_time.append(dict_emissions["duration"])

            y_pred, acc, duration_test = clf.predict(xtest=xtest, ytest=ytest)

            testing_time.append(duration_test)

        else:
            y_pred, acc, _ = clf.predict(xtest=xtest, ytest=ytest)

        ypred = ypred + y_pred

        Scores.append(acc)

    ypred = ypred / (args.runs * 1.0)
    ypred = np.argmax(ypred, axis=1)

    acc_Time = accuracy_score(y_true=ytest, y_pred=ypred, normalize=True)

    if args.track_emissions:
        df.loc[len(df)] = {
            "dataset": args.dataset,
            args.classifier + "-mean": np.mean(Scores),
            args.classifier + "-std": np.std(Scores),
            args.classifier + "Time": acc_Time,
            "Training Time-mean": np.mean(training_time),
            "Testing Time-mean": np.mean(testing_time),
            "CO2 Consumption-mean": np.mean(co2_consumption),
            "Energy Consumption-mean": np.mean(energy_consumption),
            "Country": str(dict_emissions["country_name"]),
            "Region": str(dict_emissions["region"]),
        }
    else:
        df.loc[len(df)] = {
            "dataset": args.dataset,
            args.classifier + "-mean": np.mean(Scores),
            args.classifier + "-std": np.std(Scores),
            args.classifier + "Time": acc_Time,
        }

    df.to_csv(output_directory_parent + "results_ucr.csv", index=False)
