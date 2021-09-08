#!/usr/bin/env python3
"""
Created on Thu Aug 26 20:47:33 2021

@author: anelmusic
"""

import pandas as pd


def create_title_feature(data_frame):
    data_frame["Title"] = data_frame["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())

    data_frame["Title"].replace(
        ["Mme", "Ms", "Lady", "Mlle", "the Countess", "Dona"], "Miss", inplace=True
    )
    data_frame["Title"].replace(
        ["Major", "Col", "Capt", "Don", "Sir", "Jonkheer"], "Mr", inplace=True
    )

    return data_frame


def create_ticketlen_feature(data_frame):
    data_frame["Ticket_len"] = data_frame["Ticket"].apply(lambda x: len(x))
    return data_frame


def create_ticket2letter_feature(data_frame):
    data_frame["Ticket_2letter"] = data_frame.Ticket.apply(lambda x: x[:2])
    return data_frame


def create_famsize_feature(data_frame):
    data_frame["Fam_size"] = data_frame["SibSp"] + data_frame["Parch"] + 1
    return data_frame


def create_famtype_feature(data_frame):
    # Try except because depends in fam_size feature
    data_frame["Fam_type"] = pd.cut(
        data_frame.Fam_size,
        [0, 1, 4, 7, 11],
        labels=["Solo", "Small", "Big", "Very big"],
    )
    return data_frame


def perform_feature_engineering(data_frame):
    create_title_feature(data_frame)
    create_ticketlen_feature(data_frame)
    create_ticket2letter_feature(data_frame)
    create_famsize_feature(data_frame)
    create_famtype_feature(data_frame)


perform_feature_engineering()
